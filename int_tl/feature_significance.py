import torch
from tqdm import tqdm

ce_sum_obj = torch.nn.CrossEntropyLoss(reduction='sum')


def compute_integrated_gradients(network, inputs, target_indices, baselines, m=50, create_graph=False):
    """ Gives IG approximation wrt baselines
    inputs: (num_examples, dim1, dim2 ...)
    target_indices: (num_examples, 1) --> logits of this target class will be used as the output in IG calculation
    baselines: same as inputs, or can also be just one example (1, dim1, dim2 ...)
    m: number of steps in the IG approximation
    ce_loss: if True, compute IG wrt cross entropy loss of model output. Else, use true class logit of model output
    """
    num_samples = inputs.shape[0]
    assert num_samples == target_indices.shape[0]

    diff = inputs - baselines   # this works when baseline has same shape as inputs or when it's only 1 example
    # print(f'diff.shape = {diff.shape}')

    alphas = torch.arange(m) / m     # shape: (m,)
    # print(f'alphas.shape = {alphas.shape}')
    # print(f'alphas = {alphas}')

    # This list contains a version of each example scaled by *each* m
    # Eg: 0: num_examples contains inputs scaled by m0, num_examples: 2*num_examples  contains inputs scaled by m1
    # ToDo: Optimize by making this a direct tensor operation
    scaled_inputs_list = [baselines + alpha * diff for alpha in alphas]     # shape: (m, num_samples, dim1, dim2 ...)

    scaled_inputs = torch.cat(scaled_inputs_list, dim=0)     # shape: (m * num_examples, dim1, dim2 ...)
    # print(f'scaled_inputs.shape = {scaled_inputs.shape}')
    # print(f'scaled_inputs = {scaled_inputs}')

    # Assign original label of each example to all scaled versions of itself in scaled_inputs
    expanded_target_indices = torch.cat([target_indices] * m, dim=0)     # shape: (m * num_examples, 1)
    # print(f'expanded_target_indices.shape = {expanded_target_indices.shape}')

    gradients = compute_gradients(network, scaled_inputs, expanded_target_indices, create_graph=create_graph)     # shape: (m * num_samples, dim1, dim2 ...)
    # gradients = torch.randn(m * num_samples, *inputs.shape[1:]).to(inputs.device)
    # print(f'gradients.shape = {gradients.shape}')
    # print(f'gradients = {gradients}')

    orig_shape = (m, num_samples) + (inputs.shape[1:])    # tuple: (m, num_samples, dim1, dim2 ...)
    # print(f'orig_shape = {orig_shape}')

    gradients_reshaped = gradients.reshape(orig_shape)  # shape: (m, num_samples, dim1, dim2 ...)
    # print(f'gradients_reshaped.shape = {gradients_reshaped.shape}')

    # This tensor contains the integral sum for each example
    integral_sum = torch.sum(gradients_reshaped, dim=0) / m   # shape: (num_samples, dim1, dim2 ...)
    # print(f'integral_sum.shape = {integral_sum.shape}')

    # This may not be necessary when IG is used in Robust Attribution loss
    ig_attributions = integral_sum * diff      # shape: (num_samples, dim1, dim2 ...)
    # print(f'ig_attributions.shape = {ig_attributions.shape}')

    return ig_attributions


def compute_integrated_gradients_of_loss_original(network, inputs, target_indices, baselines, m=50, create_graph=False):
    """ Gives IG approximation wrt baselines
    inputs: (num_examples, dim1, dim2 ...)
    target_indices: (num_examples, 1) --> [softmax --> CE] batch_loss of this target class will be used as the output in IG calculation
    baselines: same shape as inputs, or can also be just one example (1, dim1, dim2 ...)
    m: number of steps in the IG approximation
    """
    num_samples = inputs.shape[0]
    assert num_samples == target_indices.shape[0]

    diff = inputs - baselines   # this works when baseline has same shape as inputs or when it's only 1 example

    # Need for gradient calculation
    inputs.requires_grad_(True)
    network.zero_grad()

    softmax_func = torch.nn.Softmax(dim=1)  # dim = 1 has the logits, sum of softmaxed logits should be 1

    row_idx = torch.arange(num_samples)

    integral_sum = torch.zeros(inputs.shape, device=inputs.device)  # shape: (num_samples, dim1, dim2 ...)

    # Each step in the discrete IG approximation
    for k in range(1, m + 1):
        z = baselines + float(k) / m * diff    # shape: (num_samples, dim1, dim2 ...)
        z.requires_grad_(True)

        # Compute batch_loss for this k-th IG step
        with torch.cuda.amp.autocast(enabled=False):
            class_logits = network(z)   # shape: (num_samples, num_classes)
            softmax = softmax_func(class_logits)   # shape: (num_samples, num_classes)
            softmax_clipped = torch.clamp(softmax, min=1e-10, max=1.0)   # shape: (num_samples, num_classes)
            neg_log_softmax = -torch.log(softmax_clipped)   # shape: (num_samples, num_classes)

            # This is the neg_log_softmax value of the target class for each example
            neg_log_softmax_per_example = neg_log_softmax[row_idx, target_indices]  # shape: (num_samples, 1)

            batch_loss = torch.sum(neg_log_softmax_per_example, dim=0) / m  # Scalar

        # print(f'\t\t\tbatch_loss: {batch_loss}')

        # Compute gradients of batch_loss for this k-th IG step. Shape: (num_samples, dim1, dim2 ...)  --> shape of z
        grad = torch.autograd.grad(batch_loss, z, create_graph=create_graph, retain_graph=True)[0]

        # print(f'\t\t\t\t\tsum(batch_loss_grad): {torch.sum(grad)}')
        # print('AAAAAA - grad.shape', grad.shape)

        integral_sum = integral_sum + grad

        # print(f'\t\t\tint: {torch.sum(integral_sum)}')

    # ToDo: !!! division by m may not be correct here (taken from RRR paper) !!!
    ig_attributions = integral_sum * diff / m     # shape: (num_samples, dim1, dim2 ...)
    # ig_attributions = integral_sum * diff     # shape: (num_samples, dim1, dim2 ...)

    return ig_attributions


def compute_integrated_gradients_of_loss_refactored(network, inputs, target_indices, baselines, m=50, create_graph=False):
    assert inputs.shape[0] == target_indices.shape[0]
    diff = inputs - baselines

    # Using PyTorch's built-in CrossEntropyLoss

    global ce_sum_obj

    integral_sum = torch.zeros(inputs.shape, device=inputs.device)

    for k in range(1, m + 1):
        z = baselines + float(k) / m * diff
        z.requires_grad_(True)

        # Compute batch_loss for this k-th IG step
        batch_loss = ce_sum_obj(network(z), target_indices) / m

        # Compute gradients of batch_loss for this k-th IG step
        grad = torch.autograd.grad(batch_loss, z, create_graph=create_graph, retain_graph=True)[0]

        integral_sum = integral_sum + grad

    ig_attributions = integral_sum * diff / m
    return ig_attributions


def compute_integrated_gradients_of_loss_vectorized(network, inputs, target_indices, baselines, m=50, create_graph=False):
    """ Gives IG approximation wrt baselines
    inputs: (num_examples, dim1, dim2 ...)
    target_indices: (num_examples, 1) --> logits of this target class will be used as the output in IG calculation
    baselines: same as inputs, or can also be just one example (1, dim1, dim2 ...)
    m: number of steps in the IG approximation
    ce_loss: if True, compute IG wrt cross entropy loss of model output. Else, use true class logit of model output
    """
    num_samples = inputs.shape[0]
    assert num_samples == target_indices.shape[0]

    # device = next(network.parameters()).device
    # inputs = inputs.to(device)
    # target_indices = target_indices.to(device)
    # baselines = baselines.to(device)

    diff = inputs - baselines   # this works when baseline has same shape as inputs or when it's only 1 example
    # print(f'diff.shape = {diff.shape}')

    # alphas = torch.arange(m) / m     # shape: (m,)
    alphas = torch.arange(start=1, end=m+1) / m     # shape: (m,)

    # print(f'alphas.shape = {alphas.shape}')
    # print(f'alphas = {alphas}')

    # This list contains a version of each example scaled by *each* m
    # Eg: 0: num_examples contains inputs scaled by m0, num_examples: 2*num_examples  contains inputs scaled by m1
    # ToDo: Optimize by making this a direct tensor operation
    scaled_inputs_list = [baselines + alpha * diff for alpha in alphas]     # shape: (m, num_samples, dim1, dim2 ...)

    scaled_inputs = torch.cat(scaled_inputs_list, dim=0)     # shape: (m * num_examples, dim1, dim2 ...)
    # print(f'scaled_inputs.shape = {scaled_inputs.shape}')
    # print(f'scaled_inputs = {scaled_inputs}')

    # Assign original label of each example to all scaled versions of itself in scaled_inputs
    expanded_target_indices = torch.cat([target_indices] * m, dim=0)     # shape: (m * num_examples, 1)
    # print(f'expanded_target_indices.shape = {expanded_target_indices.shape}')

    gradients = compute_gradients_of_loss_refactored(network, scaled_inputs, expanded_target_indices, create_graph=create_graph)     # shape: (m * num_samples, dim1, dim2 ...)
    # gradients = torch.randn(m * num_samples, *inputs.shape[1:]).to(inputs.device)
    # print(f'gradients.shape = {gradients.shape}')
    # print(f'gradients = {gradients}')

    orig_shape = (m, num_samples) + (inputs.shape[1:])    # tuple: (m, num_samples, dim1, dim2 ...)
    # print(f'orig_shape = {orig_shape}')

    gradients_reshaped = gradients.reshape(orig_shape)  # shape: (m, num_samples, dim1, dim2 ...)
    # print(f'gradients_reshaped.shape = {gradients_reshaped.shape}')

    # This tensor contains the integral sum for each example
    integral_sum = torch.sum(gradients_reshaped, dim=0) / m   # shape: (num_samples, dim1, dim2 ...)
    # print(f'integral_sum.shape = {integral_sum.shape}')

    # This may not be necessary when IG is used in Robust Attribution loss
    # ig_attributions = integral_sum * diff      # shape: (num_samples, dim1, dim2 ...)

    # Additional division by m to match Chen et al. paper, training is stable this way
    ig_attributions = integral_sum * diff / m      # shape: (num_samples, dim1, dim2 ...)
    # print(f'ig_attributions.shape = {ig_attributions.shape}')

    return ig_attributions


def compute_gradients(network, inputs, target_idx, create_graph=False):
    inputs.requires_grad_(True)
    # if inputs.grad is not None:
    #     inputs.grad.zero_()
    network.zero_grad()

    with torch.cuda.amp.autocast(enabled=False):
        class_logits = network(inputs)
        # class_logits = network.forward(inputs)

    # print(f'class_logits.shape = {class_logits.shape}')
    # print(f'class_logits = {class_logits}')
    # print(f'target_idx = {target_idx}')

    row_idx = torch.arange(class_logits.shape[0])
    class_logit = class_logits[row_idx, target_idx]  # logit of the target label

    # print(f'class_logit.shape = {class_logit.shape}')
    # print(f'class_logit = {class_logit}')

    # # To get element-wise gradients of vector valued function (non-reduced loss function)
    # # ToDo: Pre-allocate and re-use tensor v
    # n = inputs.shape[0]
    # v = torch.ones(n).to(inputs.device)

    # gradients = torch.autograd.grad(class_logit, inputs, grad_outputs=v, retain_graph=True, only_inputs=True)[0]
    gradients = torch.autograd.grad(torch.unbind(class_logit), inputs, create_graph=create_graph, retain_graph=True)[0]
    # gradients = torch.randn(*inputs.shape).to(inputs.device)

    # gradients = torch.autograd.grad(class_logit, inputs)[0]
    # print(f'gradients.shape = {gradients.shape}')

    return gradients


def compute_gradients_of_loss_original(network, inputs, target_idx, create_graph=False):
    inputs.requires_grad_(True)
    # if inputs.grad is not None:
    #     inputs.grad.zero_()
    network.zero_grad()

    softmax_func = torch.nn.Softmax(dim=1)  # dim = 1 has the logits, sum of softmaxed logits should be 1

    with torch.cuda.amp.autocast(enabled=False):
        class_logits = network(inputs)
        softmax = softmax_func(class_logits)  # shape: (m * num_samples, num_classes)
        softmax_clipped = torch.clamp(softmax, min=1e-10, max=1.0)  # shape: (m * num_samples, num_classes)
        neg_log_softmax = -torch.log(softmax_clipped)  # shape: (m * num_samples, num_classes)

        row_idx = torch.arange(class_logits.shape[0])

        # This is the neg_log_softmax value of the target class for each example
        neg_log_softmax_per_example = neg_log_softmax[row_idx, target_idx]  # shape: (num_samples, 1)

        batch_loss = torch.sum(neg_log_softmax_per_example, dim=0)  # Scalar

    # print(f'class_logits.shape = {class_logits.shape}')
    # print(f'class_logits = {class_logits}')
    # print(f'target_idx = {target_idx}')

    # print(f'class_logit.shape = {class_logit.shape}')
    # print(f'class_logit = {class_logit}')

    # # To get element-wise gradients of vector valued function (non-reduced loss function)
    # # ToDo: Pre-allocate and re-use tensor v
    # n = inputs.shape[0]
    # v = torch.ones(n).to(inputs.device)

    # gradients = torch.autograd.grad(class_logit, inputs, grad_outputs=v, retain_graph=True, only_inputs=True)[0]
    gradients = torch.autograd.grad(batch_loss, inputs, create_graph=create_graph, retain_graph=True)[0]
    # gradients = torch.randn(*inputs.shape).to(inputs.device)

    # gradients = torch.autograd.grad(class_logit, inputs)[0]
    # print(f'gradients.shape = {gradients.shape}')

    return gradients


def compute_gradients_of_loss_refactored(network, inputs, target_idx, create_graph=False):
    inputs.requires_grad_(True)
    network.zero_grad()

    global ce_sum_obj

    with torch.cuda.amp.autocast(enabled=False):
        class_logits = network(inputs)
        batch_loss = ce_sum_obj(class_logits, target_idx)  # CrossEntropyLoss takes care of softmax
        gradients = torch.autograd.grad(batch_loss, inputs, create_graph=create_graph, retain_graph=True)[0]

    return gradients


def _compute_gradients_single_input(network, inputs, target_idx, softmax=False):
    # input = x.reshape(1, *x.shape)
    inputs.requires_grad = True
    # input.grad.zero_()
    network.zero_grad()

    class_logits = network(inputs)
    class_logit = class_logits[0, target_idx]  # logit of the target label

    class_logit.backward(retain_graph=True)

    gradients = inputs.grad.detach()
    return gradients


def compute_integrated_gradients_in_batches(network, inputs, target_indices, baselines, m, batch_size, progress_bar=False):
    n = len(inputs)

    ig_batches = []

    for i in tqdm(range(0, n, batch_size), disable=not progress_bar):
        inputs_batch = inputs[i: i + batch_size]
        target_indices_batch = target_indices[i: i + batch_size]

        ig_batch = compute_integrated_gradients(network, inputs_batch, target_indices_batch, baselines, m)

        ig_batches.append(ig_batch)

    # ig_batches = torch.stack(ig_batches)  # Turn the list of tensors into a single tensor with all ig maps
    ig_batches = torch.cat(ig_batches, dim=0)

    return ig_batches


def compute_gradients_in_batches(network, inputs, target_indices, batch_size, progress_bar=False):
    n = len(inputs)

    grad_batches = []

    for i in tqdm(range(0, n, batch_size), disable=not progress_bar):
        inputs_batch = inputs[i: i + batch_size]
        target_indices_batch = target_indices[i: i + batch_size]

        grad_batch = compute_gradients(network, inputs_batch, target_indices_batch)

        grad_batches.append(grad_batch)

    # grad_batches = torch.stack(grad_batches)  # Turn the list of tensors into a single tensor with all ig maps
    grad_batches = torch.cat(grad_batches, dim=0)

    return grad_batches