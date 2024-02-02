import os

import robustness.datasets
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from six.moves import urllib
from robustness.datasets import RestrictedImageNet, CustomImageNet, ImageNet
from robustness.data_augmentation import Lighting, IMAGENET_PCA, TRAIN_TRANSFORMS_IMAGENET, TEST_TRANSFORMS_IMAGENET

from utility import spurious_data_prep
from utility.data_prep_util import truncate_dataloader, preprocess_dataset, print_data_distribution, rimb_label_mapping, PerImageNormalizeTransform

from torch.utils.data import ConcatDataset, DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

loaded_datasets = {}    # To avoid loading the same dataset multiple times (in evaluate script)


def load_data(params):
    dataset = params['dataset']
    data_dir = params['data_dir']
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    transfer_type = params['transfer_type']

    if dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_preprocessed(data_dir, percent, True, batch_size, test_batch_size)
        assert params['num_classes'] == 10
    elif dataset == 'cifar10_augmented':
        train_loader, test_loader = get_cifar_data_loaders(dataset, data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'cifar10-c':
        train_loader, test_loader = get_cifar10_c_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'cifar10-p':
        train_loader, test_loader = get_cifar10_p_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'cifar100_augmented':
        train_loader, test_loader = get_cifar_data_loaders(dataset, data_dir, params)
        assert params['num_classes'] == 100
    elif dataset == 'mnist':
        train_loader, test_loader = get_mnist_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'mnist-c':
        train_loader, test_loader = get_mnist_c_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'fashion_mnist':
        train_loader, test_loader = get_fashion_mnist_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'fashion_mnist-c':
        train_loader, test_loader = get_fashion_mnist_c_preprocessed(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'flower102':
        train_loader, test_loader = get_flower102_data_loaders(data_dir, params)
        assert params['num_classes'] == 102
    elif dataset == 'svhn':
        train_loader, test_loader = get_svhn_data_loaders(data_dir, params)
        assert params['num_classes'] == 10
    elif dataset == 'restricted_imagenet':
        train_loader, test_loader = get_restricted_imagenet(data_dir, params)
        assert params['num_classes'] == 9
    elif dataset == 'restricted_imagenet_balanced':
        train_loader, test_loader = get_restricted_imagenet_balanced(data_dir, params)
        assert params['num_classes'] == 14
    elif dataset == 'rimb-c':
        train_loader, test_loader = get_rimb_c(data_dir, params)
        assert params['num_classes'] == 14
    elif dataset == 'imagenet':
        train_loader, test_loader = get_imagenet(data_dir, params)
        assert params['num_classes'] == 1000
    elif dataset == 'waterbirds':
        train_loader, test_loader = spurious_data_prep.get_waterbirds_dataloaders(data_dir, params)
        assert params['num_classes'] == 2
    elif dataset == 'celebA':
        train_loader, test_loader = spurious_data_prep.get_celebA_dataloaders(data_dir, params)
        assert params['num_classes'] == 2
    else:
        assert False

    if train_loader is not None:
        print(f'Train set no. of examples: {len(train_loader.dataset)}')
    print(f'Test set no. of examples: {len(test_loader.dataset)}')

    x, y, *metadata = test_loader.dataset.__getitem__(0)  # Get one item to get the shape of the data
    print('Shape of one example: ', x.shape)
    input_shape = params['input_shape']
    assert list(x.shape) == input_shape, f'Shape of one example {x.shape} does not match the given : {input_shape}'
    # print('y.shape: ', y.shape)

    return train_loader, test_loader


def fix_dataset_download_issue():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def get_cifar_data_loaders(dataset, data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    transfer_type = params['transfer_type']
    distributed = params['distributed_training']
    image_size = params['image_size']
    normalization = params['normalization']

    assert normalization in ['global', 'per_image']     # CIFAR must be normalized

    # CIFAR10-C is normalized with global values [we have not tried per image normalization]
    assert normalization == 'global'

    if params['pretrained']:
        assert normalization == 'global'    # Pretrained models must be normalized with global values

    image_size = 32 if image_size == -1 else image_size    # Default size is 32

    assert dataset in ['cifar10_augmented', 'cifar100_augmented']



    if dataset == 'cifar10_augmented':
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        DatasetClass = torchvision.datasets.CIFAR100

    # These are the transforms used in Salman's paper
    # train_transforms_list = [
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),    # When doing ImangeNet transfer, normalize with IMAGENET values
    #
    # ]
    # test_transforms_list = [
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    # ]

    # if transfer_type in ['from_scratch', 'full_network']:   # Do a unifying resize to 32 x 32 (see Salman et al.)
    #     print('Appending unifying resize to 32 x 32')
    #     train_transforms_list.insert(2, transforms.Resize(32))
    #     test_transforms_list.insert(2, transforms.Resize(32))

    # These are the transforms used in Singh and Mangla's paper
    train_transforms_list = [
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.25, .25, .25),    # Remove for now
        # transforms.RandomRotation(2),    # Remove for now
        transforms.ToTensor(),
    ]

    if image_size != 32:
        train_transforms_list.insert(0, transforms.Resize(image_size))  # Do this resize only if image_size != 32

    test_transforms_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if normalization == 'global':
        print('Setting (CIFAR_MEAN, CIFAR_STD) global normalization transform to train and test sets')
        train_transforms_list.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
        test_transforms_list.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    num_workers = 4
    if distributed:
        num_workers = 2

    TRAIN_TRANSFORMS = transforms.Compose(train_transforms_list)
    TEST_TRANSFORMS = transforms.Compose(test_transforms_list)

    print('TRAIN_TRANSFORMS: ', TRAIN_TRANSFORMS)
    print('TEST_TRANSFORMS: ', TEST_TRANSFORMS)

    trainset = DatasetClass(root=data_dir, train=True, download=True, transform=TRAIN_TRANSFORMS)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    testset = DatasetClass(root=data_dir, train=False, download=True, transform=TEST_TRANSFORMS)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    print_data_distribution(test_loader.dataset, 'CIFAR-10 test set distribution with preprocessing')

    return train_loader, test_loader


def get_cifar10_preprocessed(data_dir, percent=1, return_dataloaders=False, batch_size=None, test_batch_size=None):
    assert False    # Instead, use the get_cifar_data_loaders() function [dataset = 'cifar10_augmented']

    transform = transforms.Compose(
        [transforms.ToTensor(),  # Convert PIL image to tensor
         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels
         ])

    print('Preprocessing CIFAR training set')
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    preprocessed_trainset = preprocess_dataset(trainset, percent)

    print('Preprocessing CIFAR test set')
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    preprocessed_testset = preprocess_dataset(testset, percent)

    print(f'Train set shape: X = {preprocessed_trainset.tensors[0].shape}, y =  {preprocessed_trainset.tensors[1].shape}')
    print(f'Test set shape: X = {preprocessed_testset.tensors[0].shape}, y =  {preprocessed_testset.tensors[1].shape}')

    # num_workers=0 and pin_memory=True gives best performance
    if return_dataloaders:
        assert batch_size is not None and test_batch_size is not None
        preprocessed_trainset = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        preprocessed_testset = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # return trainset, testset
    return preprocessed_trainset, preprocessed_testset


""" Loads MNIST, preprocesss and load into a dataloader. Training loop has no preprocessing steps: fast """


def get_mnist_preprocessed(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    testset_only = params.get('testset_only', False)

    num_workers = 0     # Since the full preprocessed dataset is loaded into memory, no need for worker threads

    assert params['normalization'] == 'none'    # MNIST is not normalized

    print(f'Dataloader num_workers = {num_workers}')

    fix_dataset_download_issue()

    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         # transforms.Normalize((0.5), (0.5)),
         # transforms.Lambda(torch.flatten),  # Flatten image (for fully connected ANN)
         ])

    print('Preprocessing MNIST test set')
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    # print_data_distribution(testset, 'MNIST test set distribution before preprocessing')
    preprocessed_testset = preprocess_dataset(testset, percent)
    print_data_distribution(preprocessed_testset, 'MNIST test set distribution with preprocessing')
    test_loader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers)

    if testset_only:
        return None, test_loader

    print('Preprocessing MNIST training set')
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    preprocessed_trainset = preprocess_dataset(trainset, percent)
    train_loader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, test_loader
    # return preprocessed_trainset, preprocessed_testset


def get_mnist_c_preprocessed(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    testset_only = params.get('testset_only', False)

    num_workers = 0

    print(f'Dataloader num_workers = {num_workers}')

    testset = _load_mnist_c_dataset(data_dir, 'test_images.npy', 'test_labels.npy')

    print('Preprocessing MNIST test set')
    preprocessed_testset = preprocess_dataset(testset, percent)
    print_data_distribution(preprocessed_testset, 'MNIST-C test set distribution with preprocessing')
    test_loader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers)

    if testset_only:
        return None, test_loader

    trainset = _load_mnist_c_dataset(data_dir, 'train_images.npy', 'train_labels.npy')

    print('Preprocessing MNIST training set')
    preprocessed_trainset = preprocess_dataset(trainset, percent)
    train_loader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, test_loader


def get_cifar10_c_preprocessed(data_dir, params):
    percent = params['dataset_percent']
    test_batch_size = params['test_batch_size']
    severity = params['severity']
    normalization = params['normalization']
    testset_only = params.get('testset_only', False)

    assert normalization in ['global', 'per_image']    # CIFAR10-C must be normalized

    # CIFAR10-C is normalized with global values [we have not tried per image normalization]
    assert normalization == 'global'

    num_workers = 0

    print(f'Dataloader num_workers = {num_workers}')

    x_filename = params['dataset_name'] + '.npy'
    testset = _load_cifar10_c_dataset(data_dir, x_filename, 'labels.npy', severity, normalization)

    print(f'Preprocessing CIFAR10-C test set: {x_filename}')
    preprocessed_testset = preprocess_dataset(testset, percent)
    print_data_distribution(preprocessed_testset, 'CIFAR10-C test set distribution with preprocessing')
    test_loader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers)

    assert testset_only

    return None, test_loader


def get_cifar10_p_preprocessed(data_dir, params):
    percent = params['dataset_percent']
    test_batch_size = params['test_batch_size']
    severity = params['severity']
    testset_only = params.get('testset_only', False)
    normalization = params['normalization']
    testset_only = params.get('testset_only', False)

    assert normalization in ['global', 'per_image']  # CIFAR10-C must be normalized

    # CIFAR10-P is normalized with global values [we have not tried per image normalization]
    assert normalization == 'global'

    num_workers = 0

    print(f'Dataloader num_workers = {num_workers}')

    testset = _load_cifar10_p_dataset(data_dir, params['dataset_name'])

    # preprocessed_testset = preprocess_dataset(testset, percent)
    # print_data_distribution(preprocessed_testset, 'CIFAR10-P test set distribution with preprocessing')

    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    assert testset_only

    return None, test_loader


def _load_mnist_c_dataset(data_dir, x_filename, y_filename):
    X = np.load(data_dir + '/' + x_filename)  # Files saved in channels last format, eg: (10000, 28, 28, 1)
    X = np.transpose(X, (0, 3, 1, 2))   # Convert to channels first, eg: (10000, 1, 28, 28)
    X = X / 255.0  # Normalize to [0, 1]

    y = np.load(data_dir + '/' + y_filename)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)

    testset = torch.utils.data.TensorDataset(X, y)

    return testset


def _load_cifar10_c_dataset(data_dir, x_filename, y_filename, severity, normalization):
    assert 1 <= severity <= 5 or severity == -1     # -1 for all severity levels (1 to 5)

    dataset_name = f'cifar10-c-{x_filename}-severity-{severity}'

    if dataset_name in loaded_datasets:
        return loaded_datasets[dataset_name]

    X = np.load(data_dir + '/' + x_filename)  # Files saved in channels last format, eg: (10000, 32, 32, 1)
    y = np.load(data_dir + '/' + y_filename)

    assert len(X) == 50000

    start = 10000 * (severity - 1)
    end = 10000 * severity

    if severity == -1:  # All severity levels (entire set of a corrupted dataset version)
        start = 0
        end = 50000

    X = X[start: end]   # Take the appropriate data portion for the given severity
    y = y[start: end]

    # print(f'\n ****** X shape: {X.shape} ****** \n')

    # X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)

    transforms_list = [
        transforms.ToPILImage(),    # This requires that the input is a numpy array in channels last format
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),    # This converts the PIL image to a tensor in channels first format
    ]

    if normalization == 'global':
        print('Setting (CIFAR_MEAN, CIFAR_STD) global normalization transform to test set')
        transforms_list.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    test_transforms = transforms.Compose(transforms_list)

    print('Preprocessing CIFAR10-C test set')
    t0 = time.time()

    X_transformed = torch.stack([test_transforms(t) for t in X])

    print(f'Preprocessing CIFAR10-C test set complete. time taken: {time.time() - t0:.1f} secs')

    # print(f'\n ****** X_transformed.shape: {X_transformed.shape} ****** \n')

    testset = torch.utils.data.TensorDataset(X_transformed, y)

    loaded_datasets[dataset_name] = testset

    return testset


def _load_cifar10_p_dataset(data_dir, dataset_name):
    filepath = data_dir + '/' + dataset_name + '.pt'

    print(f'Loading dataset from file: {filepath}')
    tensors = torch.load(filepath)
    dataset = torch.utils.data.TensorDataset(*tensors)

    print(f'X.shape: {dataset.tensors[0].shape}')
    print(f'y.shape: {dataset.tensors[1].shape}')

    return dataset


""" Loads MNIST, preprocesss and load into a dataloader. Training loop has no preprocessing steps: fast """
def get_fashion_mnist_preprocessed(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    testset_only = params.get('testset_only', False)

    assert params['normalization'] == 'none'    # Fashion-MNIST is not normalized

    num_workers = 0

    print(f'Dataloader num_workers = {num_workers}')

    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         # transforms.Normalize((0.5), (0.5)),       # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels
         ])

    print('Preprocessing Fashion MNIST test set')
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    preprocessed_testset = preprocess_dataset(testset, percent)
    print_data_distribution(preprocessed_testset, 'Fashion MNIST test set distribution with preprocessing')
    test_loader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers)

    if testset_only:
        return None, test_loader

    print('Preprocessing Fashion MNIST training set')
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    preprocessed_trainset = preprocess_dataset(trainset, percent)
    train_loader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, test_loader
    # return preprocessed_trainset, preprocessed_testset


def get_fashion_mnist_c_preprocessed(data_dir, params):
    from datasets import load_dataset    # Huggingface datasets (pip install datasets)

    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    testset_only = params.get('testset_only', False)

    assert params['normalization'] == 'none'    # Fashion-MNIST is not normalized

    num_workers = 0

    print(f'Dataloader num_workers = {num_workers}')

    fmnist_c = load_dataset("mweiss/fashion_mnist_corrupted")

    X_test = np.array([np.array(x) for x in fmnist_c['test']['image']])
    y_test = np.array(fmnist_c['test']['label'])

    # Add channel dimension [channels first, eg: (10000, 1, 28, 28)]
    X_test = np.expand_dims(X_test, axis=1)
    X_test = X_test / 255.0  # Normalize to [0, 1]

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test)

    testset = torch.utils.data.TensorDataset(X_test, y_test)

    preprocessed_testset = preprocess_dataset(testset, percent)
    print_data_distribution(preprocessed_testset, 'Fashion MNIST Corrupted test set distribution with preprocessing')
    test_loader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers)

    if testset_only:
        return None, test_loader


def get_flower102_data_loaders(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    transfer_type = params['transfer_type']
    distributed = params['distributed_training']
    normalization = params['normalization']
    pretrained = params['pretrained']

    assert normalization in ['global', 'per_image']  # RIMB-C must be normalized

    if pretrained:
        assert normalization == 'global'  # Pretrained models must be normalized with global values

    # These are the transforms used in Salman's paper
    train_transforms_list = [
        transforms.RandomRotation(15),  # 15 degrees from Robust Attribution Regularization paper
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ]
    test_transforms_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    if normalization == 'global':
        print('Setting (IMAGENET_MEAN, IMAGENET_STD) global normalization transform to train and test sets')
        train_transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        test_transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    num_workers = 6
    if distributed:
        num_workers = 2

    # if transfer_type in ['from_scratch', 'full_network']:  # Do a unifying resize to 32 x 32
    #     train_transforms_list.insert(0, transforms.Resize(32))
    #     test_transforms_list.insert(0, transforms.Resize(32))

    TRAIN_TRANSFORMS = transforms.Compose(train_transforms_list)
    TEST_TRANSFORMS = transforms.Compose(test_transforms_list)

    trainset = torchvision.datasets.Flowers102(root=data_dir, split='train', download=True, transform=TRAIN_TRANSFORMS)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Note: there is also a validation split in this dataset

    testset = torchvision.datasets.Flowers102(root=data_dir, split='test', download=True, transform=TEST_TRANSFORMS)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers,
                                           pin_memory=True)
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers,
                                          pin_memory=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,
                                                   # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=True)

    return train_loader, test_loader


def get_svhn_data_loaders(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    transfer_type = params['transfer_type']
    distributed = params['distributed_training']
    normalization = params['normalization']

    assert normalization == 'global'    # SVHN must be normalized with global ((0.5), (0.5))

    # These are the transforms used in Salman's paper
    train_transforms_list = [
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Grayscale(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # When doing ImangeNet transfer, normalize with IMAGENET values
        transforms.Normalize((0.5), (0.5)),  # Only 1 channel to normalize when Grayscale
    ]
    test_transforms_list = [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Grayscale(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.5), (0.5)),  # Only 1 channel to normalize when Grayscale
    ]

    num_workers = 6
    if distributed:
        num_workers = 2

    if transfer_type in ['from_scratch', 'full_network']:  # Do a unifying resize to 32 x 32
        train_transforms_list.insert(0, transforms.Resize(32))
        test_transforms_list.insert(0, transforms.Resize(32))

    TRAIN_TRANSFORMS = transforms.Compose(train_transforms_list)
    TEST_TRANSFORMS = transforms.Compose(test_transforms_list)

    trainset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=TRAIN_TRANSFORMS)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=TEST_TRANSFORMS)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers,
                                           pin_memory=True)
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers,
                                          pin_memory=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,
                                                   # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=True)

    return train_loader, test_loader


def get_restricted_imagenet(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']

    num_workers = 6
    if distributed:
        num_workers = 4

    dataset = RestrictedImageNet(data_dir)
    train_loader, test_loader = dataset.make_loaders(batch_size=batch_size, val_batch_size=test_batch_size,
                                                     shuffle_train=True, shuffle_val=False, workers=num_workers)

    # Set persistent_workers true
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'{len(train_loader)} training batches, {len(test_loader)} test batches')
    print(f'{len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples')

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)
    return train_loader, test_loader


def get_restricted_imagenet_balanced(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    image_size = params['image_size']
    pretrained = params['pretrained']
    normalization = params['normalization']
    testset_only = params.get('testset_only', False)

    assert normalization in ['global', 'per_image']  # RIMB must be normalized

    if pretrained:
        assert normalization == 'global', f'pretrained = {pretrained} models must be normalized with global values'

    image_size = 224 if image_size == -1 else image_size    # Default size is 224

    num_workers = 12
    if distributed:
        num_workers = 4

    # if image_size == 224:   # Use the defaults in robustness library
    #     print('Using default transforms from robustness library')
    #     train_transform = TRAIN_TRANSFORMS_IMAGENET
    #     test_transform = TEST_TRANSFORMS_IMAGENET
    # else:   # train and test transforms copied & modified from robsutness/data_augmentation.py (to adjust size)
    #     print(f'Using custom transforms with image resizing to size: {image_size}')

    # Above block is commented out, because ...
    # we will use the following custom transform for any image size (modified from robsutness/data_augmentation.py)

    train_transform_list = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], IMAGENET_PCA['eigvec'])  # Although in robustness, we might not need this
    ]

    test_transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if normalization == 'global':
        # !!! Sunanda - not in original transform in robsutness/data_augmentation.py !!!
        print('Setting (IMAGENET_MEAN, IMAGENET_STD) global normalization transform to train and test sets')
        train_transform_list.insert(-1, transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))  # Before Lighting()
        test_transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    # ---------------------------------------------
    # Use this line when data_dir contains the full ImageNet dataset

    # dataset = CustomImageNet(data_dir, BALANCED_RANGES, transform_train=train_transform, transform_test=test_transform)

    # ---------------------------------------------
    # Use this block when data_dir contains the subset of ImageNet that is the RIMB dataset

    def map_rimb_labels(classes, class_to_idx):
        new_mapping = rimb_label_mapping    # Defined in data_prep_util
        new_classes = list(new_mapping.keys()).sort()
        return new_classes, new_mapping

    dataset = ImageNet(data_dir, transform_train=train_transform, transform_test=test_transform,
                       label_mapping=map_rimb_labels)

    # ---------------------------------------------

    train_loader, test_loader = dataset.make_loaders(batch_size=batch_size, val_batch_size=test_batch_size,
                                     only_val=testset_only, shuffle_train=True, shuffle_val=False, workers=num_workers)

    # To ensure that we have loaded the RIMB dataset correctly
    if not testset_only:
        assert len(train_loader.dataset) == 36400
    assert len(test_loader.dataset) == 1400

    # Set persistent_workers true
    if not testset_only:
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        if not testset_only:
            train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if not testset_only:
        print(f'{len(train_loader)} training batches, {len(train_loader.dataset)} training samples')

    print(f'{len(test_loader)} test batches, {len(test_loader.dataset)} test samples')

    if distributed and not testset_only:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False,  # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    if not testset_only:
        print_data_distribution(train_loader.dataset, 'RIMB train set distribution with preprocessing')

    print_data_distribution(test_loader.dataset, 'RIMB test set distribution with preprocessing')

    return train_loader, test_loader


def get_rimb_c(data_dir, params):
    testset_only = params.get('testset_only', False)
    severity = params['severity']
    dataset_name = params['dataset_name']

    noise_dir_name = os.path.basename(os.path.normpath(data_dir))


    assert dataset_name == noise_dir_name, f'dataset_name: {dataset_name}, noise_dir_name: {noise_dir_name}'

    assert testset_only


    if severity > 0:
        assert 1 <= severity <= 5  # or severity == -1     # -1 for all severity levels (1 to 5)
        full_data_dir = os.path.join(data_dir, str(severity))
        print(f'Loading RIMB-C dataset from: {data_dir}')
        assert os.path.exists(full_data_dir), f'Data directory does not exist: {full_data_dir}'
        train_loader, test_loader = get_restricted_imagenet_balanced(full_data_dir, params)
    elif severity == -1:
        print('Loading RIMB-C datasets from severity 1 to 5')
        datasets = []
        for s in range(1, 6):
            full_data_dir = os.path.join(data_dir, str(s))
            print(f'-----------> Loading RIMB-C dataset from: {full_data_dir}')
            assert os.path.exists(full_data_dir), f'Data directory does not exist: {full_data_dir}'
            train_loader, test_loader = get_restricted_imagenet_balanced(full_data_dir, params)
            datasets.append(test_loader.dataset)

        test_batch_size = params['test_batch_size']
        num_workers = 8

        combined_dataset = ConcatDataset(datasets)
        test_loader = DataLoader(combined_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=True)

        print(f'Total number of RIMB-C test samples: {len(test_loader.dataset)}')

    assert train_loader is None

    return train_loader, test_loader


def get_rimb_c_depricated(data_dir, params):
    percent = params['dataset_percent']
    test_batch_size = params['test_batch_size']
    image_size = params['image_size']
    pretrained = params['pretrained']
    testset_only = params.get('testset_only', False)
    severity = params['severity']
    normalization = params['normalization']
    dataset_name = params['dataset_name']

    assert normalization in ['global', 'per_image']  # RIMB-C must be normalized

    if pretrained:
        assert normalization == 'global'  # Pretrained models must be normalized with global values

    noise_dir_name = os.path.basename(os.path.normpath(data_dir))

    data_dir = os.path.join(data_dir, str(severity))

    assert dataset_name == noise_dir_name, f'dataset_name: {dataset_name}, noise_dir_name: {noise_dir_name}'

    assert testset_only

    assert 1 <= severity <= 5   # or severity == -1     # -1 for all severity levels (1 to 5)

    assert not pretrained, 'For ImageNet pretrained models, we MUST normalize with ImageNet mean and std'

    image_size = 224 if image_size == -1 else image_size  # Default size is 224

    num_workers = 4

    assert os.path.exists(data_dir), f'Data directory does not exist: {data_dir}'

    print(f'Loading RIMB-C dataset from: {data_dir}')

    # if image_size == 224:  # Use the defaults in robustness library
    #     print('Using default transforms from robustness library')
    #     test_transform = TEST_TRANSFORMS_IMAGENET
    # else:  # train and test transforms copied & modified from robsutness/data_augmentation.py (to adjust size)
    #     print(f'Using custom transforms with image resizing to size: {image_size}')

    # Above block is commented out, because ...
    # we will use the following custom transform for any image size (modified from robsutness/data_augmentation.py)

    test_transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if normalization == 'global':
        # !!! Sunanda - not in original transform in robsutness/data_augmentation.py !!!
        print('Setting (IMAGENET_MEAN, IMAGENET_STD) global normalization transform to test set')
        test_transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    test_transform = transforms.Compose(test_transform_list)

    dataset = torchvision.datasets.ImageFolder(data_dir, transform=test_transform)

    # Apply the label mapping to the dataset

    # def map_class_to_index(class_name):
    #     return rimb_label_mapping[class_name]
    #
    # dataset.class_to_idx = {k: map_class_to_index(k) for k, _ in dataset.class_to_idx.items()}

    dataset.class_to_idx = rimb_label_mapping

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'{len(test_loader.dataset)} test samples, {len(test_loader)} test batches')

    print_data_distribution(test_loader.dataset, 'RIMB-C (corrupted) test set distribution with preprocessing')

    return None, test_loader


def get_imagenet(data_dir, params):
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']

    num_workers = 6
    if distributed:
        num_workers = 4

    dataset = ImageNet(data_dir)
    train_loader, test_loader = dataset.make_loaders(batch_size=batch_size, val_batch_size=test_batch_size,
                                                     shuffle_train=True, shuffle_val=False, workers=num_workers)

    # Set persistent_workers true
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if percent < 1.0:
        train_loader = truncate_dataloader(train_loader, percent, batch_size, shuffle=True, num_workers=num_workers,
                                           pin_memory=True)
        test_loader = truncate_dataloader(test_loader, percent, test_batch_size, shuffle=False, num_workers=num_workers,
                                          pin_memory=True)

    print(f'{len(train_loader)} training batches, {len(test_loader)} test batches')
    print(f'{len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples')

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        batch_size_per_gpu = int(batch_size / params['num_gpus_per_node'])
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_per_gpu, shuffle=False, # Shuffling is done by sampler
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=True)

    return train_loader, test_loader


