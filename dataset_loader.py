from CONSTANTS import DIR_DATA
import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
# Source: https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5


def load(config:dict) -> (tf.data.Dataset, tf.data.Dataset):
    """
    Loads a dataset into a tf.data.Dataset object, both for training and validation
    :param config: a dictionary with keys such as:
    :return: train_dataset, val_dataset
    """

    # Create the tf.data.Datasets for training and validation
    dataset_folder = os.path.join(DIR_DATA, config["dataset"])
    train_dataset = load_dataset_split("train", os.path.join(dataset_folder, "train"), config)
    val_dataset = load_dataset_split("val", os.path.join(dataset_folder, "val"), config)
    return train_dataset, val_dataset


def load_dataset_split(split:str, directory:str, config:dict) -> tf.data.Dataset:
    """
    loads a specific split of the data from a given directory
    :param split: "train" or "val"
    :param directory: dir of the split
    :param config: a dict that must have at least the "batch_size" and "image_size" key
    :return: tf.data.Dataset
    """

    if split not in ["train", "val"]:
        raise ValueError(f"split parameter must be train or val, it is: {split}")
    dataset = image_dataset_from_directory(
        directory=directory,
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",  # means that the labels are encoded as integers
        color_mode="rgb",
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        shuffle=False,
        seed=1989
    )
    return dataset

