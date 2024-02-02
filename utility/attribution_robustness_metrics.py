from scipy.stats import spearmanr, kendalltau
import numpy as np
import torch


def compute_spearman_rank_correlation(heatmap1, heatmap2):
    """
    Both heatmaps can be of any dimension (they are flattened)
    """
    heatmap1 = heatmap1.flatten().cpu().detach().numpy()
    heatmap2 = heatmap2.flatten().cpu().detach().numpy()

    spearman_corr, _ = spearmanr(heatmap1, heatmap2)

    return spearman_corr


def compute_kendalls_correlation(heatmap1, heatmap2):
    """
    Both heatmaps can be of any dimension (they are flattened)
    """
    heatmap1 = heatmap1.flatten().cpu().detach().numpy()
    heatmap2 = heatmap2.flatten().cpu().detach().numpy()

    kendall_corr, _ = kendalltau(heatmap1, heatmap2)

    return kendall_corr


def compute_topK_intersection(heatmap1, heatmap2, K):
    """
    Top-k intersection measures intersection size of the k most important input features in the two heatmaps.
    Both heatmaps can be of any dimension (they are flattened)
    """

    heatmap1_topK = torch.argsort(heatmap1.flatten())[-K:]   # indices of top-K features in heatmap1
    heatmap2_topK = torch.argsort(heatmap2.flatten())[-K:]   # indices of top-K features in heatmap2

    heatmap1_topK = heatmap1_topK.cpu().detach().numpy()
    heatmap2_topK = heatmap2_topK.cpu().detach().numpy()

    intersection = np.intersect1d(heatmap1_topK, heatmap2_topK)     # Unique indices that are in both heatmaps

    intersection_percent = len(intersection) / K

    return intersection_percent
