from int_tl import layer_feature_significance
import torch


def compute_pgd_layer_gradient_attack(network, XX, yy, epsilon, alpha, num_iter, block_idx, sub_block_idx, random_start=True):
    """ Construct PGD IG attack examples on the examples X """

    XX_attacks = XX.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(epsilon, epsilon)
        # XX_attacks = torch.clamp(XX_attacks, min=0, max=1).detach()   # May need this step

    for t in range(num_iter):
        # print(f'compute_pgd_gradient_attack: iter {t}')
        # Set gradient-related variables
        XX.requires_grad_(True)
        XX_attacks.requires_grad_(True)

        XX_grad = layer_feature_significance.compute_layer_gradients(network, XX, yy,
                                                                     block_idx, sub_block_idx, create_graph=True)

        XX_attack_grad = layer_feature_significance.compute_layer_gradients(network, XX_attacks, yy,
                                                                            block_idx, sub_block_idx, create_graph=True)

        XX_grad_diff = XX_attack_grad - XX_grad

        XX_grad_diff_norm = torch.linalg.vector_norm(XX_grad_diff, ord=1)     # Try skipping the norm calculation

        # print(XX_grad_diff_norm)

        grad = torch.autograd.grad(XX_grad_diff_norm, XX_attacks)[0]

        # print(grad)

        # XX_attacks = XX_attacks.detach() + alpha * grad.sign()    # Try replacing with gradient values (not sign)
        XX_attacks = XX_attacks.detach() + alpha * grad.data

        # We take the full diff between original and current attack example, then we clamp it to distance budget
        delta = torch.clamp(XX_attacks - XX, min=-epsilon, max=epsilon)

        XX_attacks = XX + delta  # We may need to clamp this to [0, 1]

    return XX_attacks


def compute_layer_gradient_diff_norm_sum(network, XX, yy, XX_attacks, block_idx, sub_block_idx):
    XX_layer_grad = layer_feature_significance.compute_layer_gradients(network, XX, yy,
                                                                 block_idx, sub_block_idx, create_graph=True)

    XX_attack_layer_grad = layer_feature_significance.compute_layer_gradients(network, XX_attacks, yy,
                                                                        block_idx, sub_block_idx, create_graph=True)

    XX_layer_grad_diff = XX_attack_layer_grad - XX_layer_grad

    n = len(XX)
    XX_layer_grad_diff_vec = XX_layer_grad_diff.reshape(n, -1)  # Flatten to a set of vectors to calculate norm

    XX_layer_grad_diff_vec_norms = torch.linalg.vector_norm(XX_layer_grad_diff_vec, ord=1, dim=1)

    assert len(XX_layer_grad_diff_vec_norms) == n

    XX_layer_grad_diff_norm_sum = torch.sum(XX_layer_grad_diff_vec_norms)

    return XX_layer_grad_diff_norm_sum


def compute_layer_gradient_diff_norm_sum_in_batches(network, XX, yy, XX_attacks, block_idx, sub_block_idx, batch_size):
    n = len(XX)
    total_ig_norm_sum = 0.0

    for i in range(0, n, batch_size):
        XX_batch = XX[i: i + batch_size]
        yy_batch = yy[i: i + batch_size]
        XX_attacks_batch = XX_attacks[i: i + batch_size]
        ig_norm_sum = compute_layer_gradient_diff_norm_sum(network, XX_batch, yy_batch, XX_attacks_batch, block_idx, sub_block_idx)
        total_ig_norm_sum += ig_norm_sum

    return total_ig_norm_sum


def create_layer_gradient_attack_example(network, x, y, epsilon, alpha, num_iter, block_idx, sub_block_idx):
    """ Creates an IG attack example (x_attack)
    An IG attack will modify the input example x in an imperceptible way, but will still have a very different IG map
    x: input examples (single)
    """

    # Add new dimension (no. of samples)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)

    x_attack = compute_pgd_layer_gradient_attack(network, x, y, epsilon, alpha, num_iter, block_idx, sub_block_idx)

    x_attack = torch.squeeze(x_attack, 0)   # Remove the first dimension (no. of samples = 1)

    return x_attack
