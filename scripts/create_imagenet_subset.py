from robustness.datasets import RestrictedImageNet, CustomImageNet, CIFAR
import pandas as pd
import os, time
import shutil
from tqdm import tqdm

cifar_path = '../data'
imagenet_path = '/home/sunanda/research/Datasets/imagenet/ILSVRC/Data/CLS-LOC'

original_val_dir = 'val_original'
prepared_val_dir = 'val'
val_labels_file = '/home/sunanda/research/Datasets/imagenet/LOC_val_solution.csv'

batch_size = 512

# Class ranges for RestrictedImageNetBalanced
BALANCED_RANGES = [
    (10, 14), # birds
    (33, 37), # turtles
    (42, 46), # lizards
    (72, 76), # spiders
    (118, 122), # crabs + lobsters
    (200, 204), # some doggos
    (281, 285), # cats
    (286, 290), # big cats
    (302, 306), # beetles
    (322, 326), # butterflies
    (371, 374), # monkeys
    (393, 397), # fish
    (948, 952), # fruit
    (992, 996), # fungus
]

def prepare_validation_directory():
    print('Preparing validation directory')
    old_val_dir = imagenet_path + '/' + original_val_dir
    new_val_dir = imagenet_path + '/' + prepared_val_dir

    df = pd.read_csv(val_labels_file)

    df['Label'] = df['PredictionString'].apply(lambda x: x.split()[0])  # Extract label from second column

    val_files = os.listdir(old_val_dir)
    assert len(val_files) == 50000

    print(f'Preparing {len(val_files)} validation files')

    new_dirs = 0
    files_copied = 0
    for filename in tqdm(val_files):
        file_id = filename.split('.')[0]
        label = df[df['ImageId'] == file_id]['Label'].iloc[0]

        label_dir = new_val_dir + '/' + label

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            new_dirs += 1

        full_filename = old_val_dir + '/' + filename
        shutil.copy(full_filename, label_dir)
        files_copied += 1

    print(f'{new_dirs} new directories were created')
    print(f'{files_copied} files were copied')


def main():
    # ds = RestrictedImageNet(imagenet_path)
    ds = CustomImageNet(imagenet_path, BALANCED_RANGES)
    # ds = CIFAR(cifar_path)

    train_loader, val_loader = ds.make_loaders(batch_size=batch_size, shuffle_train=False, shuffle_val=False, workers=4)

    print(f'{len(train_loader)} training batches, {len(val_loader)} validation batches')
    print(f'~{len(train_loader) * batch_size} training samples, ~{len(val_loader) * batch_size} validation samples')

    sleep_interval = 3
    last_ts = time.time()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        time.sleep(3)
        print(f'i: {i}, time: {time.time() - last_ts}')
        last_ts = time.time()


if __name__ == '__main__':
    # prepare_validation_directory()    # Run this once to arrange validation directory into subdirs of labels
    main()

