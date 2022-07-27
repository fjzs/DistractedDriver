import os
import pandas as pd
import util
import CONSTANTS


def generate_from_original_labels(dev_fraction: float):
    """
    Creates the original dataset coming from the original labeling, with an specific dev random fraction
    :param dev_fraction: fraction to assign to a dev category (it did not have dev/val categories initially, only train)
    :return: None
    """
    # Train subfolders are the initial classes: c0, c1, ..., c9
    # Classes are 0, 1, ..., 9
    # Select only the last element of the directory tree
    directory = CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL
    subfolders = [str(f.path).split("\\")[-1] for f in os.scandir(directory) if f.is_dir()]

    # The dataframe that will store the dataset
    df = pd.DataFrame(columns=["filename", "class", "split"])

    # Append the classes file
    for subfolder in subfolders:
        subdir = os.path.join(directory, subfolder)
        files = [os.path.join(subdir,f) for f in os.listdir(subdir) if f.endswith("jpg")]  # relative path
        files = [os.path.abspath(f) for f in files]  # absolute path
        random_split = generate_random_train_dev_list(len(files), dev_fraction)
        classes = [int(subfolder[1:])]*len(files)
        mini_df = pd.DataFrame({"filename": files, "class": classes, "split": random_split})
        df = pd.concat([df, mini_df], ignore_index=True)

    # Export the final df as a .csv file
    export_to_csv(df, "datasetv00")


def sample_from_created(dataset_path:str, fraction:float, filename: str) -> None:
    """
    Creates a subset of a dataset from sampling an already existent one
    :param dataset_path: the path of the created .csv dataset
    :param fraction: fraction to be sampled [0,1]
    :param filename: name of the dataset to be created (without .csv extension)
    :return: None
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found")

    df = pd.read_csv(dataset_path)
    df = df.sample(frac=fraction)
    export_to_csv(df, filename)


def export_to_csv(df: pd.DataFrame, filename: str):
    """
    Exports a dataframe to a csv file
    :param df: the df to export
    :param filename: the filename without extension of the csv file
    :return: None
    """

    df.to_csv(os.path.join(CONSTANTS.DIR_DATA, filename+".csv"), index=False)
    print(f"Successfully generated {filename}.csv")


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


if __name__ == "__main__":
    sample_from_created(os.path.join(CONSTANTS.DIR_DATA,"datasetv00.csv"), 0.01, "datasetv00_p01")


