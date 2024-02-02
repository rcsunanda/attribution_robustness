import torch
import torch.nn as nn


class PerImageStandardization(nn.Module):
    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor image of shape (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # Note: Assuming x is of shape (B, C, H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        stddev = x.std(dim=[2, 3], keepdim=True)

        num_pixels = x.shape[2] * x.shape[3]
        min_stddev = 1.0 / torch.sqrt(torch.tensor([num_pixels]).to(x.device))

        pixel_value_scale = torch.max(stddev, min_stddev)

        return (x - mean) / pixel_value_scale

