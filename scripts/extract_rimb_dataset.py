# Command to run this script:
# python -m save_rimb_dataset

import os
import shutil
from pprint import pprint
from collections import Counter

import torch

from utility import data_prep


exp_params = {
    'dataset': 'restricted_imagenet_balanced',
    # 'data_dir': 'data/imagenet/ILSVRC/Data/CLS-LOC',    # Original ImageNet dataset path
    'data_dir': 'data/rimb/ILSVRC/Data/CLS-LOC/',    # New RIMB dataset path
    'num_classes': 14,
    'dataset_percent': 1,

    'save_dir': 'data/rimb',

    # Irrelevant params, given to prevent key errors
    'batch_size': 64,
    'test_batch_size': 64,
    'transfer_type': 'from_scratch',
    'distributed_training': 0,
    'input_shape': [3, 224, 224],
}


def save_dataset_deprecated(params):
    assert False    # This method only saves the dataset object (not the images or tensors in the dataset)

    train_loader, val_loader = data_prep.load_data(params)

    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Val set size: {len(val_loader.dataset)}')

    # Remove transforms from the datasets just in case
    train_loader.dataset.transform = None
    train_loader.dataset.transforms = None
    val_loader.dataset.transform = None
    val_loader.dataset.transforms = None

    # print(f'Train set transform: {train_loader.dataset.transform}')
    # print(f'Test set transform: {val_loader.dataset.transform}')

    print('Saving the two datasets (train and test)')

    save_dir = params['save_dir']

    assert not os.path.exists(save_dir), f'Save directory already exists: {save_dir}'

    os.mkdir(save_dir)

    torch.save(train_loader.dataset, f'{save_dir}/rimb_train.pt')
    torch.save(val_loader.dataset, f'{save_dir}/rimb_val.pt')

    print(f'Saved datasets to {save_dir}')


def load_and_test_dataset_deprecated(params):
    assert False    # This saved format does not work

    batch_size = params['batch_size']
    data_dir = params['data_dir']

    print(f'Loading the two datasets (train and test) from {data_dir}')

    # Load datasets from disk
    train_dataset = torch.load(f'{data_dir}/rimb_train.pt')
    val_dataset = torch.load(f'{data_dir}/rimb_test.pt')

    # Create data loaders from the loaded datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Val set size: {len(val_loader.dataset)}')

    print('Iterating testset')

    n = 0
    for inputs, labels in val_loader:
        print(f'inputs.shape: {inputs.shape}')
        print(f'labels.shape: {labels.shape}')

        n = n + 1
        if n == 10:
            break


def get_rimb_paths(params):
    train_loader, val_loader = data_prep.load_data(params)

    train_image_paths = [sample[0] for sample in train_loader.dataset.samples]
    test_image_paths = [sample[0] for sample in val_loader.dataset.samples]

    print(f'get_rimb_paths: Train set size: {len(train_image_paths)}')
    print(f'get_rimb_paths: Test set size: {len(test_image_paths)}')

    # print('Original Train set paths')
    # pprint(train_image_paths[0:5])
    #
    # print('Original Val set paths')
    # pprint(test_image_paths[0:5])

    return train_image_paths, test_image_paths


def copy_images(image_paths, set_name, save_dir):
    assert set_name in ['train', 'val']

    destination_paths = []

    for path in image_paths:
        start_idx = path.split(os.path.sep).index("ILSVRC")
        subpath = os.path.join(*path.split(os.path.sep)[start_idx:])
        image_dest_path = os.path.join(save_dir, subpath)

        destination_paths.append(image_dest_path)

    print(f'Destination paths for the {set_name} set. Total: {len(destination_paths)}')
    pprint(destination_paths[0:5])

    dest_subdirs = [os.path.dirname(dest) for dest in destination_paths]
    dest_subdirs_unique = set(dest_subdirs)

    # print(f'Destination subdirectories for the {set_name} set')
    # pprint(list(dest_subdirs_unique)[:10])

    # First create the necessary destination subdirectories
    print(f'Creating destination subdirectories for the {set_name} set')
    for dest_subdir in dest_subdirs_unique:
        os.makedirs(dest_subdir, exist_ok=True)

    # Copy the files to the destination paths
    print(f'Copying files for the {set_name} set')
    for source, dest in zip(image_paths, destination_paths):
        shutil.copy(source, dest)


def load_and_test_dataset(params):
    print(f'Loading RIMB from new location: {params["data_dir"]}')

    train_loader, val_loader = data_prep.load_data(params)

    print(f'load_and_test_dataset: Train set size: {len(train_loader.dataset)}')
    print(f'load_and_test_dataset: Val set size: {len(val_loader.dataset)}')

    print(f'type(train_loader.dataset): {type(train_loader.dataset)}')

    print('class_to_idx dict below')    # Useful to correctly identify the label mapping of RIMB dataset
    pprint(train_loader.dataset.class_to_idx)

    print('Iterating training set')

    n = 0
    ys = []
    for inputs, labels in train_loader:
        if n < 10:
            print(f'\tinputs.shape: {inputs.shape}')
            print(f'\tlabels.shape: {labels.shape}')

        ys.extend(labels.tolist())
        n += 1

    train_labels = [label for path, label in train_loader.dataset.samples]

    print('Train set: Unique labels & counts')
    pprint(Counter(ys))

    val_labels = [label for path, label in val_loader.dataset.samples]

    print('Val set: Unique labels & counts')
    pprint(Counter(val_labels))


def save_dataset(params):
    train_image_paths, val_image_paths = get_rimb_paths(params)

    save_dir = params['save_dir']

    assert not os.path.exists(save_dir), f'Save directory already exists: {save_dir}'
    os.mkdir(save_dir)

    copy_images(train_image_paths, 'train', save_dir)
    copy_images(val_image_paths, 'val', save_dir)


def main(params):
    # save_dataset(params)

    load_and_test_dataset(params)


if __name__ == '__main__':
    main(exp_params)