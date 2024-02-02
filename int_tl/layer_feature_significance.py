import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from utility import visualization


def compute_layer_integrated_gradients(network, inputs, target_indices, baselines, block_idx, sub_block_idx, m=50):
    '''Gives IG approximation wrt baselines
    inputs: (num_examples, dim1, dim2 ...)
    target_indices: (num_examples, 1)
    baselines: same as inputs, or can also be just one example (1, dim1, dim2 ...)
    '''
    num_samples = inputs.shape[0]
    assert num_samples == target_indices.shape[0]

    # device = next(network.parameters()).device
    # inputs = inputs.to(device)
    # target_indices = target_indices.to(device)
    # baselines = baselines.to(device)

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

    layer_gradients = compute_layer_gradients(network, scaled_inputs,
                      expanded_target_indices, block_idx, sub_block_idx)     # shape: (m * num_samples, dim1, dim2 ...)
    # layer_gradients = torch.randn(m * num_samples, *inputs.shape[1:]).to(inputs.device)
    # print(f'layer_gradients.shape = {layer_gradients.shape}')
    # print(f'layer_gradients = {layer_gradients}')

    orig_shape = (m, num_samples) + (layer_gradients.shape[1:])    # tuple: (m, num_samples, dim1, dim2 ...)
    # print(f'orig_shape = {orig_shape}')

    layer_gradients_reshaped = layer_gradients.reshape(orig_shape)  # shape: (m, num_samples, dim1, dim2 ...)
    # print(f'layer_gradients_reshaped.shape = {layer_gradients_reshaped.shape}')

    # This tensor contains the integral sum for each example
    integral_sum = torch.sum(layer_gradients_reshaped, dim=0) / m   # shape: (num_samples, dim1, dim2 ...)
    # print(f'integral_sum.shape = {integral_sum.shape}')

    # ----- Get diff tensor at layer ------
    layer_activations = None
    def _forward_hook_save(self, input, output):
        nonlocal layer_activations
        layer_activations = output

    handle = network.conv_base[block_idx][sub_block_idx].conv2.register_forward_hook(_forward_hook_save)

    dummy = network(inputs)
    inputs_layer_activations = layer_activations

    dummy = network(baselines)
    baselines_layer_activations = layer_activations

    layer_diff = inputs_layer_activations - baselines_layer_activations

    # --------------------------------------

    # print(f'layer_diff.shape = {layer_diff.shape}')

    # This may not be necessary when IG is used in Robust Attribution loss
    ig_attributions = integral_sum * layer_diff      # shape: (num_samples, dim1, dim2 ...)
    # print(f'ig_attributions.shape = {ig_attributions.shape}')

    handle.remove()

    return ig_attributions


# Differentiate model output (of a given logit) w.r.t layer activation using autograd
# block_idx is the index of the Sequential units in ResNet (in ResNet18: 4, 5, 6, 7)
# sub_block_idx is the index of the BasicBlock units within the Sequential units in ResNet (in ResNet18: 0, 1)
def compute_layer_gradients(network, inputs, target_idx, block_idx, sub_block_idx, create_graph=False, softmax=False):
    inputs.requires_grad_(True)
    network.zero_grad()

    # ResNet-18 block index check
    assert 4 <= block_idx <= 7
    assert 0 <= sub_block_idx <= 1

    # Save the layer activation tensor, so we can differentiate model output w.r.t. to it
    layer_activations_tensor = None
    def _forward_hook_save(self, input, output):
        nonlocal layer_activations_tensor
        layer_activations_tensor = output

    handle = network.conv_base[block_idx][sub_block_idx].conv2.register_forward_hook(_forward_hook_save)

    with torch.cuda.amp.autocast(enabled=False):
        class_logits = network(inputs)

    row_idx = torch.arange(class_logits.shape[0])
    class_logit = class_logits[row_idx, target_idx]  # logit of the target label

    # print('activation size:', layer_activations_tensor.size())

    gradients = torch.autograd.grad(torch.unbind(class_logit), layer_activations_tensor, create_graph=create_graph)[0]
    # print(f'autograd_gradients.shape = {gradients.shape}')

    handle.remove()

    # print('****************************')
    # print(network.conv_base[block_idx][sub_block_idx].conv2._forward_hooks)
    # print(network.conv_base[block_idx][sub_block_idx].conv2._backward_hooks)
    # print('****************************')

    return gradients


# block_idx is the index of the Sequential units in ResNet (in ResNet18: 4, 5, 6, 7)
# sub_block_idx is the index of the BasicBlock units within the Sequential units in ResNet (in ResNet18: 0, 1)
def compute_layer_gradients_with_hook(network, inputs, target_idx, block_idx, sub_block_idx, softmax=False):
    inputs.requires_grad_(True)
    network.zero_grad()

    # ResNet-18 block index check
    assert 4 <= block_idx <= 7
    assert 0 <= sub_block_idx <= 1

    layer_output_gradients = None

    def _backward_hook_save(self, grad_input, grad_output):
        nonlocal layer_output_gradients
        layer_output_gradients = grad_output[0]

    handle = network.conv_base[block_idx][sub_block_idx].conv2.register_full_backward_hook(_backward_hook_save)

    with torch.cuda.amp.autocast(enabled=False):
        class_logits = network(inputs)

    row_idx = torch.arange(class_logits.shape[0])
    class_logit = class_logits[row_idx, target_idx]  # logit of the target label

    dummy = torch.autograd.grad(torch.unbind(class_logit), inputs)[0]  # Call just to trigger the backward hook
    # print(f'hook_gradients.shape = {layer_output_gradients.shape}')

    handle.remove()

    return layer_output_gradients


def _forward_hook_print(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('--------------------------')
    print('Inside ' + self.__class__.__name__ + ' forward')

    print(f'input: {type(input)}, len: {len(input)}')
    print('input[0]: ', type(input[0]))

    print(f'output: {type(output)}, len: {len(output)}')
    print('input size:', input[0].size())
    print('output size:', output.data.size())

    print('output norm:', output.data.norm())
    print('--------------------------')


def _backward_hook_print(self, grad_input, grad_output):
    print('**************************')
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)

    print(f'grad_input: {type(grad_input)}, len: {len(grad_input)}')
    print('grad_input[0]: ', type(grad_input[0]))

    print(f'grad_output: {type(grad_output)}, len: {len(grad_output)}')
    print('grad_output[0]: ', type(grad_output[0]))
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())

    print('grad_input norm:', grad_input[0].norm())
    print('**************************')


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