import hashlib

import numpy as np
import torch
from int_tl import feature_significance


# Function to print mean, median and percentiles
def print_distribution(values):
    print(f'\tMin: {np.min(values):.2f}')
    print(f'\t1st percentile: {np.percentile(values, 1):.2f}')
    print(f'\t25th percentile: {np.percentile(values, 25):.2f}')
    print(f'\tMean: {np.mean(values):.2f}')
    print(f'\tMedian: {np.median(values):.2f}')
    print(f'\t75th percentile: {np.percentile(values, 75):.2f}')
    print(f'\t99th percentile: {np.percentile(values, 99):.2f}')
    print(f'\tMax: {np.max(values):.2f}')


def tensor_checksum(x):
    return hashlib.md5(x.contiguous().cpu().detach().numpy()).hexdigest()


class IGMapCallback:
    def __init__(self, network, dataloader):
        self.steps = 100
        self.network = network

        self.ig_maps = []

        # Save a few images for IG computation

        batch_size = 2  # Compute IG of just 2 images
        # Create new dataloader to prevent changes to the original dataloader (it's used in the training loop)
        self.dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        dataiter = iter(self.dataloader)
        # XX, yy = dataiter.next()  # A single batch
        XX, yy, *metadata = next(dataiter) # A single batch

        device = next(network.parameters()).device
        self.XX = XX.to(device)
        self.yy = yy.to(device)

        for x, y in zip(self.XX, self.yy):
            entry = {'image': x,
                    'label': y,
                    'ig_maps': {}    # key: epoch, value: IG map of img
                    }
            self.ig_maps.append(entry)

        self.baselines = torch.zeros((1, *self.XX[0].shape)).to(device)

    def on_callback(self, epoch):
        print(f'Computing IG maps for epoch {epoch}')

        XX_ig = feature_significance.compute_integrated_gradients(self.network, self.XX, self.yy, self.baselines, m=self.steps)

        for i, x_ig in enumerate(XX_ig):
            self.ig_maps[i]['ig_maps'][epoch] = x_ig
