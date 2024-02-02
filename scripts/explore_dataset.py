# Command to run this script:
# python -m scripts.explore_dataset

import torch
from collections import Counter
from pprint import pprint
from tqdm import tqdm

from utility import data_prep, visualization, common

exp_params = {
    'dataset': 'celebA',
    'data_dir': 'data',
    'num_classes': 2,
    'input_shape': [3, 224, 224],

    'dataset_percent': 0.25,   # 0.001

    'batch_size': 1,   # Important to iterate one by one
    'test_batch_size': 1,   # Important to iterate one by one

    'transfer_type': 'xxx',
    'distributed_training': 0,
    'image_size': 224,
    'normalization': 'per_image',
    'pretrained': 0,

    'random_seed': 1234567,
}

label_map = {0: 'Non-blond', 1: 'Blond'}
attribute_map = {0: 'Female', 1: 'Male'}    # metadata


def visualize_examples(examples):
    num_samples = 5

    plot1_args = []
    for i in range(num_samples):
        x = examples[i][0]
        y = examples[i][1]
        attr = examples[i][2]
        plot1_args.append({'x': x, 'title': f'{y}, {attr}'})

    visualization.plot_grid(num_samples, 1, 'Images', visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')


def main(params):
    print('Setting random seed in torch (for fixed dataset percent split)')
    torch.manual_seed(params['random_seed'])

    print('Params below:')
    pprint(params)

    print('Loading dataset')
    train_loader, test_loader = data_prep.load_data(params)

    torch.set_printoptions(profile="short")

    print('\n======= Exploring train_loader =======\n')
    explore_dataloader(train_loader)

    print('\n======= Exploring test_loader =======\n')
    explore_dataloader(test_loader)

    # visualize_examples(examples)


def explore_dataloader(dataloader):
    examples = []   # list of (input, label_name, attribute_name, group_key) tuples

    print('Iterating dataloader, len(dataloader):', len(dataloader))

    for inputs, labels, *metadata in tqdm(dataloader):
        # Important, because we are taking the only element in the batch using index 0 below
        assert len(inputs) == 1  # !!! DO NOT REMOVE THIS ASSERTION !!!

        # Get the metadata element from the above iterator's tuple
        metadata = metadata[0]

        # print(f'inputs.shape: {inputs.shape}')
        # print(f'labels.shape: {labels.shape}')
        # print(f'metadata.shape: {metadata.shape}')

        # Get the only element in the inputs and labels batch
        x = inputs[0]
        y = labels[0].item()

        # Get the only element in the metadata batch
        metadata_tensor = metadata[0]   # (shape: [3])

        # Note: The metadata tensor of an example has 3 elements: [attribute, _, _]. Take the first element
        # So take the first element as the attribute

        attr = metadata_tensor[0].item()

        group_key = y * 2 + attr

        y_name = label_map[y]
        attr_name = attribute_map[attr]

        examples.append((x, y_name, attr_name, group_key))

    # [print(f'y: {y_name}, attr: {attr_name}') for (x, y_name, attr_name) in examples]

    print(f'No of examples: {len(examples)}')

    group_keys = [k for (x, y_name, attr_name, k) in examples]

    # Count the no. of examples per group key using a Counter
    group_key_counts = Counter(group_keys)

    print(f'group_key_counts below:')
    pprint(group_key_counts)


if __name__ == '__main__':
    main(exp_params)