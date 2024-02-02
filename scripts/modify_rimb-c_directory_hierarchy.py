import os
import shutil

root_dir = "data/rimb-c-hendrycks"
new_root_dir = "data/rimb-c-hendrycks-hierarchy"


def main():
    assert os.path.exists(root_dir)

    for noise_type in os.listdir(root_dir):     # Traverse the root directory
        noise_type_dir = os.path.join(root_dir, noise_type)

        assert os.path.isdir(noise_type_dir)   # We do not expect intermediate files in this hierarchy

        for noise_severity in os.listdir(noise_type_dir):   # Traverse the noise severity directories within the noise type directory
            noise_severity_dir = os.path.join(noise_type_dir, noise_severity)

            assert os.path.isdir(noise_severity_dir)  # We do not expect intermediate files in this hierarchy

            new_dir = os.path.join(new_root_dir, noise_type, noise_severity, 'test')

            os.makedirs(new_dir, exist_ok=True)

            # Traverse the noise severity directories to get at the class directories
            for class_name in os.listdir(noise_severity_dir):
                class_dir = os.path.join(noise_severity_dir, class_name)
                shutil.move(class_dir, new_dir)


if __name__ == '__main__':
    main()
