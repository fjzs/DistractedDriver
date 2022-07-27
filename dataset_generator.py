import os
import util
from CONSTANTS import DIR_DATA, DIR_ORIGINAL_TRAINVAL
import shutil


def generate_from_original_labels(total_fraction:float, val_fraction: float) -> None:
    """
    Creates the original dataset coming from the original labeling, with specific fractions of total selection and
    val splitting.
    :param total_fraction: fraction of the whole trainval dataset to consider
    :param val_fraction: fraction to assign to a val category (it did not have val categories initially, only train)
    :return: None
    """
    if not (0 < total_fraction <= 1):
        raise ValueError("total_fraction parameter incorrect")
    if not (0 < val_fraction <= 1):
        raise ValueError("val_fraction parameter incorrect")

    # Create the new data_xxx fraction folder within data if non existant
    folder_name = "data_" + str(int(total_fraction*100)).zfill(3)
    folder_path = os.path.join(DIR_DATA, folder_name)
    folder_path_train = os.path.join(folder_path, "train")
    folder_path_val = os.path.join(folder_path, "val")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(folder_path_train):
        os.mkdir(folder_path_train)
    if not os.path.isdir(folder_path_val):
        os.mkdir(folder_path_val)

    # Read the classes names c0, c1, ..., c9
    classes_names = [str(f.path).split("\\")[-1] for f in os.scandir(DIR_ORIGINAL_TRAINVAL) if f.is_dir()]
    for class_name in classes_names:
        print(f"Class: {class_name}")

        # Check if dir is needed:
        folder_path_train_class_name = os.path.join(folder_path_train, class_name)
        folder_path_val_class_name = os.path.join(folder_path_val, class_name)
        if not os.path.isdir(folder_path_train_class_name):
            os.mkdir(folder_path_train_class_name)
        if not os.path.isdir(folder_path_val_class_name):
            os.mkdir(folder_path_val_class_name)

        # Read the files from the original location
        subdir = os.path.join(DIR_ORIGINAL_TRAINVAL, class_name)
        files = [f for f in os.listdir(subdir) if f.endswith("jpg")]  # relative path
        util.shuffle(files)

        # Select a sample of the files
        files = files[0:int(len(files)*total_fraction)]
        num_val_files = int(len(files)*val_fraction)

        # Copy the files from the original location to the destination location
        for i,f in enumerate(files):
            source = os.path.join(subdir, f)
            if i < num_val_files:  # Copy into the val folder
                destination = os.path.join(folder_path_val_class_name, f)
                shutil.copy(source, destination)
            else:  # Copy into the train folder
                destination = os.path.join(folder_path_train_class_name, f)
                shutil.copy(source, destination)
        print(f"\t# val images: {num_val_files}")
        print(f"\t# train images: {len(files) - num_val_files}")




