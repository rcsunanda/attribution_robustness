from int_tl import feature_significance
from utility import attribution_robustness_metrics
import torch


def compute_topk_ifia_attack(network, XX, yy, epsilon, alpha, num_iter, ig_steps, k_top, random_start=True):
    """ Implements the tok-k Iterative Feature Importance Attacks (IFIA) from Ghorbani et al.
    For the saliency map, we use Integrated Gradients (IG) (any valid saliency map method can be used)
    """

    assert len(XX) == 1, 'Only single image supported'

    # Use minimum and maximum pixel values in the given batch for clipping pixel values of generated attacks
    min_value = torch.min(XX)
    max_value = torch.max(XX)

    # --------- Compute initial variable needed for top-k algorithm ---------
    XX_attacks = XX.clone().detach()
    XX_final_attack = XX.clone().detach()

    # Compute IG of original image
    zero_baseline = torch.zeros((1, *XX[0].shape)).to(XX.device)
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines=zero_baseline, m=ig_steps)

    XX_ig = XX_ig * ig_steps    # Testing

    # Sum across channels dimension (first dim = no. of samples == 1, second dim = channels) and normalize the heatmap
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)
    XX_ig = XX_ig / torch.sum(XX_ig)  # Normalize the heatmap

    with torch.no_grad():
        _, original_y_pred = network(XX).max(1)

    # print(f'XX.shape: {XX.shape}')
    # print(f'XX_ig.shape: {XX_ig.shape}')

    # Get indices of top-k important pixels
    sorted_idx = torch.argsort(XX_ig.flatten())[-k_top:]
    topk_mask = torch.zeros(XX_ig.numel(), device=XX.device)
    topk_mask[sorted_idx] = 1

    min_criteria = 1.0

    if random_start:
        # Starting at a uniformly random point
        XX_attacks = XX_attacks + torch.empty_like(XX_attacks).uniform_(-epsilon, epsilon)
        XX_attacks.requires_grad_(True)
        # XX_attacks = torch.clamp(XX_attacks, min=0, max=1).detach()   # May need this step

    for t in range(num_iter):
        # print(f'compute_pgd_attack: iter {t}')
        # Set gradient-related variables
        XX_attacks.requires_grad_(True)

        XX_attacks_ig = feature_significance.compute_integrated_gradients(network, XX_attacks, yy,
                                                               baselines=zero_baseline, m=ig_steps, create_graph=True)

        # Sum across channels and normalize the heatmap
        XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)
        XX_attacks_ig = XX_attacks_ig / torch.sum(XX_attacks_ig)

        # ------- Top-k direction ---------------

        topK_loss = -torch.sum(XX_attacks_ig.flatten() * topk_mask)
        grad = torch.autograd.grad(topK_loss, XX_attacks)[0]

        # print(topK_loss.item())

        # ---------------------------------------
        # Update attack example (PGD)

        XX_attacks = XX_attacks.detach() + alpha * grad.sign()

        # We take the full diff between original and current attack example, then we clamp it to distance budget (PGD)
        delta = torch.clamp(XX_attacks - XX, min=-epsilon, max=epsilon)

        XX_attacks = XX + delta  # We may need to clamp this to [0, 1]

        # Clamp to ensure valid pixel range
        XX_attacks = torch.clamp(XX_attacks, min=min_value, max=max_value)

        # ---------------------------------------
        # Select attack with minimum criterion that also matches the prediction label to the original label

        with torch.no_grad():
            _, attack_y_pred = network(XX_attacks).max(1)
            # print(f'y_true: {y}, attack_y_pred: {attack_y_pred[0]}, original_y_pred: {original_y_pred[0]}')

        if attack_y_pred[0] == original_y_pred[0]:
            # intersection = attribution_robustness_metrics.compute_topK_intersection(XX_ig, XX_attacks_ig, k_top)
            kendall_corr = attribution_robustness_metrics.compute_kendalls_correlation(XX_ig, XX_attacks_ig)

            if kendall_corr < min_criteria:
                min_criteria = kendall_corr
                XX_final_attack = XX_attacks

        # # !!! REMOVE !!!
        # # After commenting out the above block for testing, return the final attack after num_iter iterations
        # XX_final_attack = XX_attacks

    return XX_final_attack


def create_topk_ifia_attack(network, x, y, epsilon, alpha, num_iter, adv_ig_steps, k_top, random_start=True):
    """ Creates an IG attack example (x_attack)
    An IG attack will modify the input example x in an imperceptible way, but will still have a very different IG map
    x: input examples (single)
    """

    # Add new dimension (no. of samples)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)

    x_attack = compute_topk_ifia_attack(network, x, y, epsilon, alpha, num_iter, adv_ig_steps, k_top, random_start)

    x_attack = torch.squeeze(x_attack, 0)   # Remove the first dimension (no. of samples = 1)

    return x_attack
