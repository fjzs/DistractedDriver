import os
import util
import CONSTANTS
import shutil


def generate_from_original_labels(val_fraction: float):
    """
    Creates the original dataset coming from the original labeling, with an specific val random fraction
    :param val_fraction: fraction to assign to a dev category (it did not have dev/val categories initially, only train)
    :return: None
    """
    # Train subfolders are the initial classes: c0, c1, ..., c9
    # Classes are 0, 1, ..., 9
    # Select only the last element of the directory tree
    source_dir = CONSTANTS.DIR_ORIGINAL_TRAINVAL
    dest_dir_train = os.path.join(CONSTANTS.DIR_DATA,"train")
    dest_dir_val = os.path.join(CONSTANTS.DIR_DATA,"val")
    subfolders = [str(f.path).split("\\")[-1] for f in os.scandir(source_dir) if f.is_dir()]

    for subfolder in subfolders:
        print(f"Subfolder: {subfolder}")
        subdir = os.path.join(source_dir, subfolder)
        files = [f for f in os.listdir(subdir) if f.endswith("jpg")]  # relative path
        util.shuffle(files)
        num_val_files = int(len(files)*val_fraction)
        for i,f in enumerate(files):
            source = os.path.join(subdir, f)
            if i < num_val_files:  # Copy into the val folder
                destination = os.path.join(dest_dir_val, subfolder, f)
                shutil.copy(source, destination)
            else:  # Copy into the train folder
                destination = os.path.join(dest_dir_train, subfolder, f)
                shutil.copy(source, destination)
        print(f"\t# val images: {num_val_files}")
        print(f"\t# train images: {len(files) - num_val_files}")


def generate_random_train_dev_list(size:int, dev_fraction: float) -> list:
    """
    generates a list with a random order of train and dev tags
    :param size: size of the list
    :param dev_fraction: fraction of dev appearances
    :return:
    """
    train_size = int(size*(1-dev_fraction))
    dev_size = size - train_size
    train_dev_list = ["train"]*train_size + ["dev"]*dev_size
    util.shuffle(train_dev_list)
    return train_dev_list




