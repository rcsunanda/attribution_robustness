from int_tl import feature_significance
import torch


def compute_pgd_gradient_attack(network, XX, yy, epsilon, alpha, num_iter, random_start=True):
    """ Construct PGD IG attack examples on the examples X """

    XX_attacks = XX.clone().detach()
    # XX_attacks = XX.clone()

    if random_start:
        # Starting at a uniformly random point
        XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(epsilon, epsilon)
        # XX_attacks = torch.clamp(XX_attacks, min=0, max=1).detach()   # May need this step

    create_graph = True

    for t in range(num_iter):
        print(f'compute_pgd_gradient_attack: iter {t}')
        # Set gradient-related variables
        XX.requires_grad_(True)
        XX_attacks.requires_grad_(True)

        XX_grad = feature_significance.compute_gradients(network, XX, yy, create_graph=None)
        XX_attack_grad = feature_significance.compute_gradients(network, XX_attacks, yy, create_graph=create_graph)

        XX_grad.requires_grad_(True)
        XX_attack_grad.requires_grad_(True)

        print(f'XX_grad.requires_grad: {XX_grad.requires_grad}')
        print(f'XX_attack_grad.requires_grad: {XX_attack_grad.requires_grad}')

        XX_grad_diff = XX_attack_grad - XX_grad

        print(f'XX_grad_diff.requires_grad: {XX_grad_diff.requires_grad}')

        XX_grad_diff_norm = torch.linalg.vector_norm(XX_grad_diff, ord=1)     # Try skipping the norm calculation

        print(XX_grad_diff_norm)

        grad = torch.autograd.grad(XX_grad_diff_norm, XX_attacks)[0]

        # print(grad)

        # XX_attacks = XX_attacks.detach() + alpha * grad.sign()    # Try replacing with gradient values (not sign)
        XX_attacks = XX_attacks.detach() + alpha * grad.data

        # We take the full diff between original and current attack example, then we clamp it to distance budget
        delta = torch.clamp(XX_attacks - XX, min=-epsilon, max=epsilon)

        XX_attacks = XX + delta  # We may need to clamp this to [0, 1]

        create_graph = False

    return XX_attacks


def compute_gradient_diff_norm_sum(network, XX, yy, XX_attacks):
    XX_grad = feature_significance.compute_gradients(network, XX, yy, create_graph=True)
    XX_attack_grad = feature_significance.compute_gradients(network, XX_attacks, yy, create_graph=True)

    XX_grad_diff = XX_attack_grad - XX_grad

    n = len(XX)
    XX_grad_diff_vec = XX_grad_diff.reshape(n, -1)     # Flatten to a set of vectors to calculate norm

    XX_grad_diff_vec_norms = torch.linalg.vector_norm(XX_grad_diff_vec, ord=1, dim=1)

    assert len(XX_grad_diff_vec_norms) == n

    XX_grad_diff_norm_sum = torch.sum(XX_grad_diff_vec_norms)

    return XX_grad_diff_norm_sum


def compute_gradient_diff_norm_sum_in_batches(network, XX, yy, XX_attacks, batch_size):
    n = len(XX)
    total_ig_norm_sum = 0.0

    for i in range(0, n, batch_size):
        XX_batch = XX[i: i + batch_size]
        yy_batch = yy[i: i + batch_size]
        XX_attacks_batch = XX_attacks[i: i + batch_size]
        ig_norm_sum = compute_gradient_diff_norm_sum(network, XX_batch, yy_batch, XX_attacks_batch)
        total_ig_norm_sum += ig_norm_sum

    return total_ig_norm_sum


def create_gradient_attack_example(network, x, y, epsilon, alpha, num_iter):
    """ Creates an IG attack example (x_attack)
    An IG attack will modify the input example x in an imperceptible way, but will still have a very different IG map
    x: input examples (single)
    """

    # Add new dimension (no. of samples)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)

    x_attack = compute_pgd_gradient_attack(network, x, y, epsilon, alpha, num_iter)

    x_attack = torch.squeeze(x_attack, 0)   # Remove the first dimension (no. of samples = 1)

    return x_attack
