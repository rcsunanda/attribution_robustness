# Run the following command:
# python -m preprocess_cifar-p_dataset

import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np


from pprint import pprint
import time

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

data_dir = 'data/CIFAR-10-P'
save_dir = 'data/CIFAR-10-P-preprocessed'

filenames = [
    'brightness.npy',
    'gaussian_noise.npy',
    'gaussian_noise_2.npy',
    'gaussian_noise_3.npy',
    'motion_blur.npy',
    'rotate.npy',
    'scale.npy',
    'shear.npy',
    'shot_noise.npy',
    'shot_noise_2.npy',
    'shot_noise_3.npy',
    'snow.npy',
    'spatter.npy',
    'speckle_noise.npy',
    'speckle_noise_2.npy',
    'speckle_noise_3.npy',
    'tilt.npy',
    'translate.npy',
    'zoom_blur.npy',
]


def preprocess_and_save_all_cifar10_p_datasets():
    # Load the test set labels of the original CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    y = next(iter(testloader))[1].numpy()

    for filename in filenames:
        print(f'\n------------ Preprocessing CIFAR10-P dataset: {filename} ------------')
        t0 = time.time()

        dataset = preprocess_cifar10_p_dataset(data_dir, filename, y)

        print(f'X.shape: {dataset.tensors[0].shape}')
        print(f'y.shape: {dataset.tensors[1].shape}')

        print(f'Saving CIFAR10-P dataset: {filename}')

        save_dataset(dataset, save_dir, filename)

        print(f'total time taken: {time.time() - t0:.1f} secs')


def preprocess_cifar10_p_dataset(data_dir, x_filename, y):
    # Files saved in channels last format, eg: (10000, num_permutations+1, 32, 32, 1)
    # The num_permutations+1 in the second axis represent the clean CIFAR-10 dataset + 30 perturbed versions
    X = np.load(data_dir + '/' + x_filename)

    # print(f'\n------ original X.shape: {X.shape} -----\n')

    X = X[:, 1:, :, :, :]   # Drop the clean CIFAR-10 version. New shape: (10000, num_permutations, 32, 32, 1)
    num_permutations = X.shape[1]
    X = X.reshape(X.shape[0] * num_permutations, 32, 32, 3)   # Put all permutations together New shape: (300000, 32, 32, 3)

    # print(f'\n------ reshaped X.shape: {X.shape} -----\n')

    # print(f'\n------ original y.shape: {y.shape} -----\n')

    y = y.repeat(num_permutations)    # Repeat the labels 30 times element wise to match the new X shape

    # print(f'\n------ reshaped y.shape: {y.shape} -----\n')

    y = torch.from_numpy(y)

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),    # This requires that the input is a numpy array in channels last format
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),    # This converts the PIL image to a tensor in channels first format
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    X_transformed = torch.stack([test_transforms(t) for t in X])

    # print(f'\n ****** X_transformed.shape: {X_transformed.shape} ****** \n')

    testset = torch.utils.data.TensorDataset(X_transformed, y)

    return testset


def save_dataset(dataset, save_dir, filename):
    fname = filename.replace('.npy', '.pt')
    save_filepath = save_dir + '/' + fname
    torch.save(dataset.tensors, save_filepath)


def load_dataset(filepath):
    print(f'Loading dataset from file: {filepath}')
    tensors = torch.load(filepath)
    dataset = torch.utils.data.TensorDataset(*tensors)

    print(f'X.shape: {dataset.tensors[0].shape}')
    print(f'y.shape: {dataset.tensors[1].shape}')

    return dataset


if __name__ == '__main__':
    preprocess_and_save_all_cifar10_p_datasets()

    # Test loading the dataset
    # dataset = load_dataset('data/CIFAR-10-P-preprocessed/brightness.pt')
