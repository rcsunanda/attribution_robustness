from int_tl import layer_feature_significance
import torch


def compute_pgd_attack(network, XX, yy, block_idx, sub_block_idx, epsilon, alpha, num_iter, ig_steps, random_start=True):
    """ Construct PGD IG attack examples on the examples X """

    XX_attacks = XX.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(epsilon, epsilon)
        # XX_attacks = torch.clamp(XX_attacks, min=0, max=1).detach()   # May need this step

    for t in range(num_iter):
        # print(f'compute_pgd_attack: iter {t}')
        # Set gradient-related variables
        XX_attacks.requires_grad_(True)

        layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, XX_attacks,
                                                                              block_idx, sub_block_idx, m=ig_steps)

        layer_ig_norm = torch.linalg.vector_norm(layer_ig, ord=1)     # Try skipping the norm calculation

        # print(layer_ig_norm)

        grad = torch.autograd.grad(layer_ig_norm, XX_attacks)[0]

        # print(grad)

        # XX_attacks = XX_attacks.detach() + alpha * grad.sign()    # Try replacing with gradient values (not sign)
        XX_attacks = XX_attacks.detach() + alpha * grad.data

        # We take the full diff between original and current attack example, then we clamp it to distance budget
        delta = torch.clamp(XX_attacks - XX, min=-epsilon, max=epsilon)

        XX_attacks = XX + delta  # We may need to clamp this to [0, 1]

    return XX_attacks


def compute_layer_ig_norm_sum(network, XX, yy, XX_attacks, block_idx, sub_block_idx, norm_ig_steps):
    XX_attacks_layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, XX_attacks,
                                                                           block_idx, sub_block_idx, m=norm_ig_steps)

    n = len(XX)
    attacks_layer_ig_vecs = XX_attacks_layer_ig.reshape(n, -1)     # Flatten to a set of vectors to calculate norm

    attacks_layer_ig_vec_norms = torch.linalg.vector_norm(attacks_layer_ig_vecs, ord=1, dim=1)

    assert len(attacks_layer_ig_vec_norms) == n

    layer_ig_norm_sum = torch.sum(attacks_layer_ig_vec_norms)

    return layer_ig_norm_sum


def compute_layer_ig_norm_sum_in_batches(network, XX, yy, XX_attacks, block_idx, sub_block_idx, norm_ig_steps, batch_size):
    n = len(XX)
    total_ig_norm_sum = 0.0

    for i in range(0, n, batch_size):
        XX_batch = XX[i: i + batch_size]
        yy_batch = yy[i: i + batch_size]
        XX_attacks_batch = XX_attacks[i: i + batch_size]
        ig_norm_sum = compute_layer_ig_norm_sum(network, XX_batch, yy_batch, XX_attacks_batch,
                                                block_idx, sub_block_idx, norm_ig_steps)
        total_ig_norm_sum += ig_norm_sum

    return total_ig_norm_sum


def create_layer_ig_attack_example(network, x, y, block_idx, sub_block_idx, epsilon, alpha, num_iter, adv_ig_steps):
    """ Creates an IG attack example (x_attack)
    An IG attack will modify the input example x in an imperceptible way, but will still have a very different IG map
    x: input examples (single)
    """

    # Add new dimension (no. of samples)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)

    x_attack = compute_pgd_attack(network, x, y, block_idx, sub_block_idx, epsilon, alpha, num_iter, adv_ig_steps)

    x_attack = torch.squeeze(x_attack, 0)   # Remove the first dimension (no. of samples = 1)

    return x_attack
