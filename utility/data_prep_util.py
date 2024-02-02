import numpy as np
import torch
from PIL import Image

from utility import misc

# The following mapping was extracted from loading the RIMB dataset correctly with the full ImageNet dataset
rimb_label_mapping = {
     'n01530575': 0,
     'n01537544': 0,
     'n01664065': 1,
     'n01669191': 1,
     'n01687978': 2,
     'n01693334': 2,
     'n01773157': 3,
     'n01774750': 3,
     'n01978287': 4,
     'n01983481': 4,
     'n02097474': 5,
     'n02098413': 5,
     'n02123045': 6,
     'n02124075': 6,
     'n02125311': 7,
     'n02128925': 7,
     'n02167151': 8,
     'n02174001': 8,
     'n02277742': 9,
     'n02281787': 9,
     'n02486261': 10,
     'n02488291': 10,
     'n02607072': 11,
     'n02655020': 11,
     'n07742313': 12,
     'n07753113': 12,
     'n12998815': 13,
     'n13052670': 13
}


class PerImageNormalizeTransform(object):
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # Calculate mean and std along the (C, H, W) dimensions

        mean = tensor.mean(dim=[1, 2], keepdim=True)
        stddev = tensor.std(dim=[1, 2], keepdim=True)

        num_pixels = tensor[0].numel()  # Assuming a single image tensor with first dimension as channels
        min_stddev = 1.0 / torch.sqrt(torch.tensor([num_pixels]))

        pixel_value_scale = torch.max(stddev, min_stddev)

        return (tensor - mean) / pixel_value_scale


class InMemoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, fixed_transform=None, random_transform=None):
        assert data.shape[0] == targets.shape[0]

        # Copied from PyTorch CIFAR: doing this so that it is consistent with all other datasets to return a PIL Image
        # Do the PIL conversion upfront to save compute during training
        self.data = []
        for elem in data:
            img = Image.fromarray(elem)
            if fixed_transform is not None:   # Do any fixed_transforms (deterministic) upfront to save compute during training
                img = fixed_transform(img)
            self.data.append(img)

        self.targets = targets
        self.random_transform = random_transform

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]

        if self.random_transform is not None:
            data = self.random_transform(data)

        return data, target

    def __len__(self):
        # return self.data.shape[0]
        return len(self.data)


def create_in_memory_dataset(dataset, fixed_transform, random_transform, percent):
    dataset_size = int(np.ceil(dataset.data.shape[0] * percent))  # percent will give a single batch that contains full dataset

    indices = list(np.arange(dataset_size))

    X_all = dataset.data[indices]
    y_all = np.array(dataset.targets)[indices]

    assert X_all.shape[0] == dataset_size  # Verify that this single batch contains the full dataset

    preprocessed_dataset = InMemoryImageDataset(X_all, y_all, fixed_transform, random_transform)
    return preprocessed_dataset


def truncate_dataloader(dataloader, percent, batch_size, shuffle, num_workers, pin_memory):
    truncated_size = int(percent * len(dataloader.dataset))
    remaining = len(dataloader.dataset) - truncated_size

    # print(f'Transform before truncating: {dataloader.dataset.transform}')

    truncated_dataset, _ = torch.utils.data.random_split(dataloader.dataset, [truncated_size, remaining])

    truncated_loader = torch.utils.data.DataLoader(truncated_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    # print(f'Transform after truncating: {truncated_loader.dataset.dataset.transform}')

    return truncated_loader


def preprocess_dataset(dataset, percent):
    batch_size = int(np.ceil(len(dataset) * percent))  # percent will give a single batch that contains full dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    # x_preprocessed, y = dataiter.next()  # A single batch that contains full dataset
    x_preprocessed, y = next(dataiter)  # A single batch that contains full dataset

    assert x_preprocessed.shape[0] == batch_size  # Verify that this single batch contains the full dataset

    preprocessed_dataset = torch.utils.data.TensorDataset(x_preprocessed, y)
    return preprocessed_dataset


def print_data_distribution(dataset, prefix):
    batch_size = 256     # A small batch is enough to get an idea about the distribution

    print(f'------ {prefix}, batch_size: {batch_size} --------')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    # X, y = dataiter.next()  # A single batch
    X, y, *metadata = next(dataiter)  # A single batch

    X = np.array(X).flatten()

    misc.print_distribution(X)
    print(f'------------------------')
