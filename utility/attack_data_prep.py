import os

import numpy as np
import torch
import torchvision.transforms as transforms


def load_attack_data(params):
    if not params['load_attack_data']:
        print('Not loading attack data')
        return None, None

    dataset = params['dataset']
    data_dir = params['attack_data_dir']

    if dataset == 'mnist':
        natural_loader, attacks_loader = get_mnist_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'fashion_mnist':
        return None, None
        natural_loader, attacks_loader = get_fashion_mnist_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'cifar10_augmented':
        natural_loader, attacks_loader = get_cifar_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'waterbirds':
        # !! We are using the same data loading function as CIFAR10 ---> this could be one function for all datasets !!!
        natural_loader, attacks_loader = get_generic_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 2
    elif dataset == 'celebA':
        natural_loader, attacks_loader = get_generic_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 2
    elif dataset == 'restricted_imagenet_balanced':
        # !! We are using the same data loading function as CIFAR10 ---> this could be one function for all datasets !!!
        natural_loader, attacks_loader = get_generic_attacks_preprocessed(data_dir, params)
        assert params['num_classes'] == 14
    else:
        assert False, f'Unknown dataset: {dataset}'

    print(f'Attack set no. of natural examples: {len(natural_loader.dataset)}')
    print(f'Attack set no. of attack examples: {len(attacks_loader.dataset)}')

    return natural_loader, attacks_loader


def _load_dataset_from_disk(directory):
    assert os.path.exists(directory), f'Dataset directory does not exist: {directory}'

    print(f'Loading attack dataset from directory: {directory}')

    # Load data directly to CPU regardless of which device it was saved from
    XX = torch.load(f'{directory}/XX.pt', map_location=torch.device('cpu'))
    labels = torch.load(f'{directory}/labels.pt', map_location=torch.device('cpu'))
    XX_attacks = torch.load(f'{directory}/XX_attacks.pt', map_location=torch.device('cpu'))

    # To prevent autograd errors when iterating dataloader.
    # requires_grad should be set only in places where gradient calculation is needed
    XX.requires_grad_(False)
    XX_attacks.requires_grad_(False)

    print(f'\tXX.shape: {XX.shape}')
    print(f'\tXX_attacks.shape: {XX_attacks.shape}')
    print(f'\tlabels.shape: {labels.shape}')

    return XX, labels, XX_attacks


def get_mnist_attacks_preprocessed(data_dir, params):
    percent = params.get('attack_dataset_percent', params['dataset_percent'])
    batch_size = params.get('attack_batch_size', params['test_batch_size'])

    batch_size = params['test_batch_size']

    num_workers = 0

    XX, labels, XX_attacks = _load_dataset_from_disk(data_dir)

    # Take a subset of the dataset
    n = int(np.ceil(XX.shape[0] * percent))
    XX = XX[:n]
    labels = labels[:n]
    XX_attacks = XX_attacks[:n]

    natural_dataset = torch.utils.data.TensorDataset(XX, labels)
    natural_loader = torch.utils.data.DataLoader(natural_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    attacks_dataset = torch.utils.data.TensorDataset(XX_attacks, labels)
    attacks_loader = torch.utils.data.DataLoader(attacks_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return natural_loader, attacks_loader


def get_cifar_attacks_preprocessed(data_dir, params):
    percent = params.get('attack_dataset_percent', params['dataset_percent'])
    batch_size = params['attack_batch_size']

    if batch_size < 0:  # Default param value of -1
        batch_size = params['test_batch_size']

    num_workers = 0

    XX, labels, XX_attacks = _load_dataset_from_disk(data_dir)

    # Take a subset of the dataset
    n = int(np.ceil(XX.shape[0] * percent))
    XX = XX[:n]
    labels = labels[:n]
    XX_attacks = XX_attacks[:n]

    natural_dataset = torch.utils.data.TensorDataset(XX, labels)
    natural_loader = torch.utils.data.DataLoader(natural_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    attacks_dataset = torch.utils.data.TensorDataset(XX_attacks, labels)
    attacks_loader = torch.utils.data.DataLoader(attacks_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return natural_loader, attacks_loader


def get_generic_attacks_preprocessed(data_dir, params):
    percent = params.get('attack_dataset_percent', params['dataset_percent'])
    batch_size = params['attack_batch_size']

    if batch_size < 0:  # Default param value of -1
        batch_size = params['test_batch_size']

    num_workers = 0

    XX, labels, XX_attacks = _load_dataset_from_disk(data_dir)

    # Take a subset of the dataset
    n = int(np.ceil(XX.shape[0] * percent))
    XX = XX[:n]
    labels = labels[:n]
    XX_attacks = XX_attacks[:n]

    image_size = params['image_size']
    if XX.shape[1] != image_size:
        print('Applying transforms to attack data')

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])

        # Apply the transforms to X and XX_attacks
        XX_transformed = torch.stack([transform(x) for x in XX])
        XX_attacks_transformed = torch.stack([transform(x) for x in XX_attacks])

        XX = XX_transformed
        XX_attacks = XX_attacks_transformed

    natural_dataset = torch.utils.data.TensorDataset(XX, labels)
    natural_loader = torch.utils.data.DataLoader(natural_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    attacks_dataset = torch.utils.data.TensorDataset(XX_attacks, labels)
    attacks_loader = torch.utils.data.DataLoader(attacks_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return natural_loader, attacks_loader
