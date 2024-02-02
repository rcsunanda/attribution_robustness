# Command to run this script:
# python -m scripts.extract_rimb-c_dataset

import os
import shutil
from tqdm import tqdm

from utility import data_prep_util


exp_params = {
    'data_dir': 'data/imagenet-c',    # Location of the ImageNet-C dataset
    'save_dir': 'data/rimb-c',

    # Specific noise subdirectories to extract (remove this param to extract all)
    # 'specific_noise_subdirs': ['gaussian_noise'],
}


def copy_rimb_images_in_dir(severity_dir, save_dir):
    class_dirs = [f.path for f in os.scandir(severity_dir) if f.is_dir()]

    existing_dirs = set()   # Keep track of existing dirs to avoid many OS calls

    rimb_label_mapping = data_prep_util.rimb_label_mapping  # key: class_dir_name, value: rimb_label

    saved_class_dir_names = set()

    for class_dir in class_dirs:
        class_dir_name = os.path.basename(os.path.normpath(class_dir))

        if class_dir_name in rimb_label_mapping:
            save_class_dir = os.path.join(save_dir, class_dir_name)

            if save_class_dir not in existing_dirs:
                os.makedirs(save_class_dir, exist_ok=True)
                existing_dirs.add(save_class_dir)

            shutil.copytree(class_dir, save_class_dir, dirs_exist_ok=True)  # Overwrites will occur due to dirs_exist_ok

            saved_class_dir_names.add(class_dir_name)

    # print(f'Following class directories were extracted: {saved_class_dir_names}')

    # Verify that correct subset was copied (both ways)

    for class_dir_name in saved_class_dir_names:
        assert class_dir_name in rimb_label_mapping, f'Class dir name not in rimb_label_mapping: {class_dir_name}'

    for rimb_class_dir_name, rimb_label in rimb_label_mapping.items():
        assert rimb_class_dir_name in saved_class_dir_names, f'RIMB class dir name not in imagenet-c: {rimb_class_dir_name}'


def save_dataset(params):
    data_dir = params['data_dir']
    save_root_dir = params['save_dir']
    specific_noise_subdirs = params.get('specific_noise_subdirs', [])

    noise_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    print('Following noise subdirectories found')
    [print(f'\t{subdir}') for subdir in noise_dirs]

    for noise_dir in tqdm(noise_dirs):
        noise_subdir_name = os.path.basename(os.path.normpath(noise_dir))
        if specific_noise_subdirs and noise_subdir_name not in specific_noise_subdirs:
            continue

        print(f'Copying images in subdirectory: {noise_dir}')

        severity_dirs = [f.path for f in os.scandir(noise_dir) if f.is_dir()]

        for severity_dir in severity_dirs:
            # print(f'\tSource dir: {severity_dir}')

            components = os.path.normpath(severity_dir).split(os.path.sep)
            assert len(components) >= 2  # Must have the form data_dir/noise_dir/severity_dir
            destination_subdir = os.path.sep.join(components[-2:])  # /noise_dir/severity_dir

            # save_root_dir/noise_dir/severity_dir/test/
            destination_dir = os.path.join(save_root_dir, destination_subdir, 'test')

            # print(f'\tDestination dir: {destination_dir}')

            print(f'\tCopy: {severity_dir} --> {destination_dir}')

            copy_rimb_images_in_dir(severity_dir, destination_dir)


def main(params):
    save_dataset(params)

    # load_and_test_dataset(params)


if __name__ == '__main__':
    main(exp_params)