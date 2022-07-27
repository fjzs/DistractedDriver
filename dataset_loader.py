import CONSTANTS
import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
# Source: https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5


def load(config:dict):
    """
    Loads a dataset into a tf.data.Dataset object, both for training and validation
    :param config: a dictionary with keys such as:
    :return: train_dataset, val_dataset
    """

    # Create the tf.data.Datasets for training and validation
    train_dir = os.path.join(CONSTANTS.DIR_DATA, "train")
    train_dataset = load_dataset_split("train", train_dir, config)
    val_dir = os.path.join(CONSTANTS.DIR_DATA, "val")
    val_dataset = load_dataset_split("val", val_dir, config)

    return train_dataset, val_dataset


def load_dataset_split(split:str, directory:str, config:dict) -> tf.data.Dataset:

    if split not in ["train", "val"]:
        raise ValueError(f"split parameter must be train or val, it is: {split}")
    dataset = image_dataset_from_directory(
        directory=directory,
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",  # labels are encoded as integers
        color_mode="rgb",
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        shuffle=True,
        seed=1989
    )
    return dataset

