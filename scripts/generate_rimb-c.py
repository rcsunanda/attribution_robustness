# Command to run this script:
# python -m generate_rimb-c

import os
import shutil
from pprint import pprint
from collections import Counter
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import imagenet_c

from utility import data_prep

exp_params = {
    'dataset': 'restricted_imagenet_balanced',
    'data_dir': 'data/rimb/ILSVRC/Data/CLS-LOC/',    # RIMB dataset path
    'dataset_percent': 1,

    'save_dir': 'data/rimb-c',

    # Corruption params
    'corruptions': ['gaussian_noise'],
    'severities': [1],

    # Irrelevant params, given to prevent key errors
    'num_classes': 14,
    'batch_size': 64,
    'test_batch_size': 1,
    'transfer_type': 'from_scratch',
    'distributed_training': 0,
    'input_shape': [3, 224, 224],
    'image_size': 224,
    'pretrained': 0,
}


def corrupt_rimb(params, courruption, severity):
    assert params['test_batch_size'] == 1, 'test_batch_size must be 1 to ensure we iterate one by one'
    assert params['image_size'] == 224, 'image_size must be 224 for the following transforms to be valid'
    assert params['dataset_percent'] == 1, 'dataset_percent must be 1 to avoid path issues of truncated dataset'

    save_dir = params['save_dir']
    save_subdir = os.path.join(save_dir, courruption, str(severity))

    assert not os.path.exists(save_subdir), f'Save subdirectory already exists: {save_subdir}'

    _, val_loader = data_prep.load_data(params)

    # We want to ensure that the val_loader is not shuffled so that it matches with the order of other lists
    assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler), \
        f'Bad val_loader sampler: {type(val_loader.sampler)}'

    val_image_paths = [sample[0] for sample in val_loader.dataset.samples]

    print(f'load_and_test_dataset: Val set size: {len(val_loader.dataset)}')
    print(f'val_image_paths: {len(val_image_paths)}')

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    final_save_dirs = set()  # Save the final save dirs to avoid many mkdir calls

    for image_path in tqdm(val_image_paths):
        assert os.path.exists(image_path), f'Image path does not exist: {image_path}'

        image = Image.open(image_path)
        image = image.convert('RGB')
        # x = np.array(image)

        image = transform(image)

        x_corrupt = imagenet_c.corrupt(image, severity=severity, corruption_name=courruption)

        # print(f'x.shape: {x.shape}, x_corrupt.shape: {x_corrupt.shape}'

        class_dir = os.path.basename(os.path.dirname(image_path))
        final_save_dir = os.path.join(save_subdir, class_dir)

        filename = os.path.basename(image_path)
        save_path = os.path.join(final_save_dir, filename)

        if final_save_dir not in final_save_dirs:
            os.makedirs(final_save_dir, exist_ok=True)
            final_save_dirs.add(final_save_dir)

        Image.fromarray(np.uint8(x_corrupt)).save(save_path, quality=85, optimize=True)


    # print('Iterating val set')

    # for inputs, labels in val_loader:
    #     assert len(inputs) == 1
    #     x = inputs[0]
    #     y = labels[0]
    #
    #     x = x.detach().numpy().transpose((1, 2, 0))     # Convert to channels last format
    #
    #     x_corrupt = imagenet_c.corrupt(x, severity=1, corruption_name=courruption)
    #
    #     print(f'x.shape: {x.shape}, x_corrupt.shape: {x_corrupt.shape}')



def main(params):
    courriptions = params['corruptions']
    severities = params['severities']

    for courruption in courriptions:
        for severity in severities:
            print(f'Preparing corrupted dataset for courruption: {courruption}, severity: {severity}')

            corrupt_rimb(params, courruption, severity)

    # load_and_test_dataset(params)


if __name__ == '__main__':
    main(exp_params)