from CONSTANTS import DIR_DATA, DIR_ORIGINAL_TRAINVAL, FILEPATH_DRIVER_FILE_SPLIT
import os
import pandas as pd
import shutil
import util


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


def generate_from_csv(fraction:float) -> None:
    # TODO: add the test split
    if not (0 < fraction <= 1):
        raise ValueError("total_fraction parameter incorrect")

    # Create the new data_xxx fraction folder within data if non existant
    folder_name = "data_noleak_" + str(int(fraction * 100)).zfill(3)
    folder_path = os.path.join(DIR_DATA, folder_name)
    folder_path_train = os.path.join(folder_path, "train")
    folder_path_val = os.path.join(folder_path, "val")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(folder_path_train):
        os.mkdir(folder_path_train)
    if not os.path.isdir(folder_path_val):
        os.mkdir(folder_path_val)

    # Read the .csv file where the split of each file is indicated
    # Sample rows:
    # subject_id	img_file	    split
    # p002	        img_100057.jpg	val
    # p002	        img_100116.jpg	val
    df = pd.read_csv(FILEPATH_DRIVER_FILE_SPLIT, sep=";")

    # Select the rows from the dataframe sample a fraction
    df_train_class = df.loc[df['split'] == "train"]
    df_val_class = df.loc[df['split'] == "val"]
    df_train_class_sample = df_train_class.sample(frac=fraction, random_state=1989).reset_index()
    df_val_class_sample = df_val_class.sample(frac=fraction, random_state=1989).reset_index()
    df_train_val_sample = pd.concat([df_train_class_sample, df_val_class_sample])

    # Read the classes names c0, c1, ..., c9
    classes_names = [str(f.path).split("\\")[-1] for f in os.scandir(DIR_ORIGINAL_TRAINVAL) if f.is_dir()]
    for class_name in classes_names:
        class_code = class_name[0:2]
        print(f"Class: {class_name} and code: {class_code}")
        # Check if dir is needed to be created at destination:
        folder_path_train_class_name = os.path.join(folder_path_train, class_code)
        folder_path_val_class_name = os.path.join(folder_path_val, class_code)
        if not os.path.isdir(folder_path_train_class_name):
            os.mkdir(folder_path_train_class_name)
        if not os.path.isdir(folder_path_val_class_name):
            os.mkdir(folder_path_val_class_name)

        # Read the files from the class folder
        subdir = os.path.join(DIR_ORIGINAL_TRAINVAL, class_name)
        class_files = [f for f in os.listdir(subdir) if f.endswith("jpg")]  # relative path

        # Copy the files from the original location to the destination location
        train_num = 0
        val_num = 0
        for f in class_files:
            split = df_train_val_sample.loc[df_train_val_sample['img_file'] == f]["split"]
            if split.empty:
                continue
            else:
                split = str(split.values[0])
            filepath_origin = os.path.join(subdir, f)
            filepath_destination = None
            if split == "train":
                filepath_destination = os.path.join(folder_path_train_class_name, f)
                train_num += 1
            elif split == "val":
                filepath_destination = os.path.join(folder_path_val_class_name, f)
                val_num += 1
            else:
                raise ValueError("split not recognized")
            shutil.copy(filepath_origin, filepath_destination)

        print(f"\tTrain images: {train_num}")
        print(f"\tVal images: {val_num}")


def generate_test_from_existing_val(dataset_name) -> None:
    """

    :param dataset_name: such as "data_no_leak_100"
    :return:
    """

    # Create the new test folder within the dataset folder
    folder_path = os.path.join(DIR_DATA, dataset_name)
    folder_path_val = os.path.join(folder_path, "val")
    folder_path_test = os.path.join(folder_path, "test")

    if not os.path.isdir(folder_path):
        raise ValueError(f"folder not existent: {folder_path}")
    if not os.path.isdir(folder_path_test):
        os.mkdir(folder_path_test)

    # Read the .csv file where the split of each file is indicated
    # Sample rows:
    # subject_id	img_file	    split
    # p002	        img_100057.jpg	val
    # p002	        img_100116.jpg	val
    df = pd.read_csv(FILEPATH_DRIVER_FILE_SPLIT, sep=";")

    # Select the rows from the dataframe sample a fraction
    test_files = df.loc[df['split'] == "test"]["img_file"].to_list()

    # Read the classes names c0, c1, ..., c9
    classes_names = [str(f.path).split("\\")[-1] for f in os.scandir(folder_path_val) if f.is_dir()]
    for class_name in classes_names:

        print(f"Working in class: {class_name}")

        # Check if dir is needed to be created at destination:
        folder_path_test_class_name = os.path.join(folder_path_test, class_name)
        if not os.path.isdir(folder_path_test_class_name):
            os.mkdir(folder_path_test_class_name)

        # Read the files from the val/class folder
        folder_path_val_class = os.path.join(folder_path_val, class_name)
        val_class_files = [f for f in os.listdir(folder_path_val_class) if f.endswith("jpg")]  # relative path
        test_class_files = [f for f in val_class_files if f in test_files]

        # Cut the files from the original val location to the destination test location
        print(f"About to cut {len(test_class_files)} from val to test")
        for f in test_class_files:
            filepath_origin = os.path.join(folder_path_val_class, f)
            filepath_destination = os.path.join(folder_path_test_class_name, f)
            shutil.copy(filepath_origin, filepath_destination)
            os.remove(filepath_origin)


def sample_from_existing_dataset(fraction_train: float, fraction_val: float, fraction_test: float) -> None:
    """

    :param fraction_train:
    :param fraction_val:
    :param fraction_test:
    :return:
    """

    # Create the new dataset folder
    source_folder_path = os.path.join(DIR_DATA, "data_noleak_100")
    destination_folder_name = "data_noleak_" + str(int(fraction_train * 100)).zfill(3)
    destination_folder_path = os.path.join(DIR_DATA, destination_folder_name)
    if not os.path.isdir(destination_folder_path):
        os.mkdir(destination_folder_path)

    splits = ["train", "val", "test"]
    frac_per_split = {"train": fraction_train, "val": fraction_val, "test": fraction_test}
    classes = ["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
    if not os.path.isdir(source_folder_path):
        raise ValueError(f"folder not existent: {source_folder_path}")

    for split in splits:
        print(f"Working in split: {split}")
        fraction = frac_per_split[split]
        source_folder_split_path = os.path.join(source_folder_path, split)
        destination_folder_split_path = os.path.join(destination_folder_path, split)
        if not os.path.isdir(destination_folder_split_path):
            os.mkdir(destination_folder_split_path)

        for class_name in classes:
            print(f"\tWorking in class: {class_name}")
            source_folder_split_class_path = os.path.join(source_folder_split_path, class_name)
            destination_folder_split_class_path = os.path.join(destination_folder_split_path, class_name)
            if not os.path.isdir(destination_folder_split_class_path):
                os.mkdir(destination_folder_split_class_path)

            files = [f for f in os.listdir(source_folder_split_class_path) if f.endswith("jpg")]  # relative path
            util.shuffle_list(files)
            sample_size = int(fraction*len(files))
            files = files[0:sample_size]

            # Now copy these files to the destination folder
            for f in files:
                filepath_origin = os.path.join(source_folder_split_class_path, f)
                filepath_destination = os.path.join(destination_folder_split_class_path, f)
                shutil.copy(filepath_origin, filepath_destination)


