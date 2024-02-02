from int_tl import feature_significance
from utility import misc
import torch


def compute_pgd_attacks(network, XX, yy, epsilon, alpha, num_iter, ig_steps, random_start=True):
    """ Construct PGD IG attack examples on the examples X """

    assert random_start is True, 'Without random_start, IG norm is 0.0 for all iterations --> attack does not change'

    # Use minimum and maximum pixel values in the given batch for clipping pixel values of generated attacks
    min_value = torch.min(XX)
    max_value = torch.max(XX)
    # print(f'-----> min_value: {min_value}, max_value: {max_value}')

    XX_attacks = XX.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(-epsilon, epsilon)
        XX_attacks.requires_grad_(True)

        # XX_attacks = torch.clamp(XX_attacks, min=min_value, max=max_value).detach()   # May need this step

        # rand_delta = (epsilon + epsilon) * torch.rand(XX_attacks.shape).to(XX_attacks.device) - epsilon
        # XX_attacks = XX_attacks + rand_delta

    for t in range(num_iter):
        # print(f'compute_pgd_attack: iter {t}')
        # Set gradient-related variables
        XX_attacks.requires_grad_(True)

        # XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=True)
        XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=True)
        # XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss_vectorized(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=False)

        XX_ig_norm = torch.linalg.vector_norm(XX_XX_attack_ig, dim=None, ord=1)

        # print(f't: {t}, XX_ig_norm: {XX_ig_norm}')

        grad = torch.autograd.grad(XX_ig_norm, XX_attacks)[0]

        # print(f'compute_pgd_attacks. grad_sum: {torch.sum(grad)}')

        # print(f'grad_sign_positives_over_negative: {torch.sum(grad.sign())}, '
        #       f'percent: {torch.sum(grad.sign()) * 100 / grad.numel()} %')

        XX_attacks_preclamp = XX_attacks.detach() + alpha * grad.sign()
        # XX_attacks_preclamp = XX_attacks.detach() + alpha * grad.data

        # We take the full diff between original and current attack example, then we clamp it to distance budget
        delta = torch.clamp(XX_attacks_preclamp - XX, min=-epsilon, max=epsilon)
        XX_attacks = XX + delta

        # Alternative to taking delta
        # XX_attacks = torch.clamp(XX_attacks, min=XX - epsilon, max=XX + epsilon)

        # Clamp to ensure valid pixel range
        XX_attacks = torch.clamp(XX_attacks, min=min_value, max=max_value)

        # # Debug prints
        # print(f'\n------------ t: {t} -------------')
        # print(f'\t XX_XX_attack_ig checksum: {misc.tensor_checksum(XX_XX_attack_ig)}')
        # print(f'\t XX_ig_norm checksum: {misc.tensor_checksum(XX_ig_norm)}')
        # print(f'\t XX_ig_norm: {XX_ig_norm}')
        # print(f'\t XX_ig_sum: {XX_ig_sum}')
        # # print(f'\t XX_ig_norm_manual checksum: {misc.tensor_checksum(XX_ig_norm_manual)}')
        # # print(f'\t XX_ig_norm_manual: {XX_ig_norm_manual}')
        # print(f'\t grad checksum: {misc.tensor_checksum(grad)}')
        # print(f'\t delta checksum: {misc.tensor_checksum(delta)}')
        # print(f'\t XX_attacks_preclamp checksum: {misc.tensor_checksum(XX_attacks_preclamp)}')
        # print(f'\t XX_attacks checksum: {misc.tensor_checksum(XX_attacks)}')

    return XX_attacks


def compute_one_step_attacks(network, XX, yy, epsilon, alpha, ig_steps):
    """ Construct PGD IG attack examples on the examples X
        This version only takes one step of PGD (no iterations). We expect this to be faster, but empirically it is not
        Also, there is no clamping to epsilon. random starting is done (no option to disable it)
    """

    # Use minimum and maximum pixel values in the given batch for clipping pixel values of generated attacks
    min_value = torch.min(XX)
    max_value = torch.max(XX)
    # print(f'-----> min_value: {min_value}, max_value: {max_value}')

    XX_attacks = XX.clone().detach()

    # Starting at a uniformly random point within the distance bowl
    XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(-epsilon, epsilon)
    XX_attacks.requires_grad_(True)

    # -----------
    # Compute the IG (from the original image to the attack image)
    # XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=True)
    XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=True)
    # XX_XX_attack_ig = feature_significance.compute_integrated_gradients_of_loss_vectorized(network, XX, yy, baselines=XX_attacks, m=ig_steps, create_graph=False)

    XX_ig_norm = torch.linalg.vector_norm(XX_XX_attack_ig, dim=None, ord=1)

    grad = torch.autograd.grad(XX_ig_norm, XX_attacks)[0]

    XX_attacks = XX_attacks.detach() + alpha * grad.sign()  # !!! is detach() needed here? !!!

    XX_attacks = torch.clamp(XX_attacks, min=min_value, max=max_value)  # To ensure valid pixel range

    # -----------

    return XX_attacks


def compute_ig_norm_sum(network, XX, yy, XX_attacks, norm_ig_steps):
    XX_XX_attacks_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines=XX_attacks,
                                                                                 m=norm_ig_steps, create_graph=True)

    # XX_XX_attacks_ig = feature_significance.compute_integrated_gradients_of_loss_vectorized(network, XX, yy, baselines=XX_attacks,
    #                                                                              m=norm_ig_steps, create_graph=False)

    XX_ig_norm_sum = torch.linalg.vector_norm(XX_XX_attacks_ig, dim=None, ord=1)

    return XX_ig_norm_sum


def compute_grad_diff_norm(network, XX, yy, XX_attacks):
    XX_grad = feature_significance.compute_gradients_of_loss_refactored(network, XX, yy, create_graph=True)

    XX_attacks_grad = feature_significance.compute_gradients_of_loss_refactored(network, XX_attacks, yy, create_graph=True)

    XX_grad_diff_norm = torch.linalg.vector_norm(XX_grad - XX_attacks_grad, dim=None, ord=1)

    return XX_grad_diff_norm
