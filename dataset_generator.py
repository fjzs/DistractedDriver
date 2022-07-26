import os
import pandas as pd
import util
import CONSTANTS

def generate_from_original_labels(dev_fraction: float, directory: str = CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL):
    # Train subfolders are the initial classes: c0, c1, ..., c9
    # Classes are 0, 1, ..., 9
    # Select only the last element of the directory tree
    subfolders = [str(f.path).split("\\")[-1] for f in os.scandir(directory) if f.is_dir()]

    # The dataframe that will store the dataset
    df = pd.DataFrame(columns=["file", "class", "split"])

    # Append the classes file
    for subfolder in subfolders:
        subdir = os.path.join(directory, subfolder)
        files = [os.path.join(subdir,f) for f in os.listdir(subdir) if f.endswith("jpg")]
        random_split = generate_random_train_dev_list(len(files), dev_fraction)
        classes = [int(subfolder[1:])]*len(files)
        mini_df = pd.DataFrame({"file": files, "class": classes, "split": random_split})
        df = df.append(mini_df)

    # Export the final df as a .csv file
    df.to_csv(os.path.join(CONSTANTS.DIR_DATA, "datasetv00.csv"))






def generate_random_train_dev_list(size:int, dev_fraction: float) -> list:
    train_size = int(size*(1-dev_fraction))
    dev_size = size - train_size
    train_dev_list = ["train"]*train_size + ["dev"]*dev_size
    util.shuffle(train_dev_list)
    return train_dev_list




