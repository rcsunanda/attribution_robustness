from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

import torch
import torchvision.transforms as transforms

from utility.data_prep_util import truncate_dataloader, preprocess_dataset, print_data_distribution

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_waterbirds_dataloaders(data_dir, params):
    image_size = params['image_size']
    image_size = 224 if image_size == -1 else image_size    # Default size is 224
    return _get_wilds_dataloaders('waterbirds', params, size=image_size, num_workers=4)


def get_celebA_dataloaders(data_dir, params):
    image_size = params['image_size']
    image_size = 224 if image_size == -1 else image_size  # Default size is 224
    return _get_wilds_dataloaders('celebA', params, size=image_size, num_workers=4)


def _get_wilds_dataloaders(dataset_name, params, size, num_workers):
    root_dir = params['data_dir']
    percent = params['dataset_percent']
    batch_size = params['batch_size']
    test_batch_size = params['test_batch_size']
    distributed = params['distributed_training']
    normalization = params['normalization']
    pretrained = params['pretrained']

    assert normalization in ['global', 'per_image']  # WILDS datasets (Waterbirds, CelebA) must be normalized

    if pretrained:
        if normalization != 'global':
            print('\n\t!!!!!!!! WARNING: Pretrained models must be normalized with global values !!!!!!!!\n')
        # assert normalization == 'global'  # Pretrained models must be normalized with global values

    if distributed:
        num_workers = max(2, num_workers // 2)

    # Load the full dataset, and download it if necessary
    # Can specify a data directory with "root_dir='xxx'"
    dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)

    train_transforms_list = [
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.75, 4./3.)),    # Paper transform
        #### transforms.RandomCrop(32, padding=4), --> only for CIFAR-10
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    test_transforms_list = [
        transforms.Resize(size),    # 32 for CIFAR, 256 for ImageNet, 224 for CelebA & Waterbirds
        transforms.CenterCrop(size),    # 32 for CIFAR, 224 for ImageNet, 224 for CelebA & Waterbirds
        transforms.ToTensor(),
    ]

    if normalization == 'global':
        print('Setting (IMAGENET_MEAN, IMAGENET_STD) global normalization transform to train and test sets')
        train_transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        test_transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    TRAIN_TRANSFORMS = transforms.Compose(train_transforms_list)
    TEST_TRANSFORMS = transforms.Compose(test_transforms_list)

    trainset = dataset.get_subset("train", transform=TRAIN_TRANSFORMS)   # Extract the training set
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Prepare the standard data loader. Note: Use this if you want to do things like grouping
    # train_loader = get_train_loader("standard", train_data, batch_size=batch_size)

    testset = dataset.get_subset("test", transform=TEST_TRANSFORMS)   # Extract the test set
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

    print_data_distribution(test_loader.dataset, f'{dataset_name} test set distribution with preprocessing')

    return train_loader, test_loader

