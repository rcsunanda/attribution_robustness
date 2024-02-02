import os
import sys
import math
import ast
import time

imagenet_folder = '/home/sunanda/Datasets/imagenet/ILSVRC/Data/CLS-LOC'
imagenet_train_folder = f'{imagenet_folder}/train'

num_subsets = 5


def get_subset_list_from_cmd_args():
    num_args = len(sys.argv)
    if num_args != 2:
        print('Usage:')
        print('python tar_imagenet_train_set [0, 1, 2]')
        sys.exit(1)

    subset_list = sys.argv[1]
    subset_list = ast.literal_eval(subset_list)

    return subset_list


def main():
    subset_list = get_subset_list_from_cmd_args()

    print(f'Starting compressing subsets: {subset_list}. Total subsets: {num_subsets}')

    subdirs = os.listdir(imagenet_train_folder)

    # subdirs = subdirs[0:25]

    num_subdirs = len(subdirs)
    subset_size = math.ceil(num_subdirs / num_subsets)

    print(f'No. of subdirectories: {num_subdirs}, subset_size: {subset_size}')


    print(f'subset_list: {subset_list}')

    t0 = time.time()

    for idx in subset_list:
        t_start = time.time()

        start = idx * subset_size
        end = (idx+1) * subset_size

        subset_i = subdirs[start:end]

        print(f'Subset {idx}: {num_subdirs}, start: {start}, end: {end}, subset_i_size: {len(subset_i)}')
        print(f'Compressing subset {idx}')

        subset_i_paths = [os.path.join('train', file) for file in subset_i]

        subset_i_paths_str = ' '.join(subset_i_paths)

        print(f'Subset start: {subset_i_paths[0]}')
        print(f'Subset end: {subset_i_paths[-1]}')

        tar_file = f'train_{idx}.tar.gz'

        command = f'cd {imagenet_folder}; ' \
                  f'tar -c --use-compress-program="pigz -p 8" -f {tar_file} {subset_i_paths_str}'

        # print(command)

        os.system(command)

        time_taken = (time.time() - t_start) / 60
        print(f'Finished compressing subset {idx}. Time taken: {time_taken:.1f} mins')

    time_taken = (time.time() - t0) / 60
    print(f'Finished compressing subsets: {subset_list}. Total subsets: {num_subsets}. Time taken: {time_taken:.1f} mins')


if __name__ == '__main__':
    main()
