import os
import tensorflow as tf
import util
from keras.utils import image_dataset_from_directory
# Source: https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5


def load_dataset_split(split:str, config:dict, shuffle: bool) -> tf.data.Dataset:
    """
    loads a specific split of the data from a given directory
    :param split: "train" or "val"
    :param config: configuration parameters
    :param shuffle: Use True when training, False for error analysis
    :return: tf.data.Dataset
    """

    if split not in ["train", "val"]:
        raise ValueError(f"split parameter must be train or val, it is: {split}")

    batch_size = config["batch_size"] if split == "train" else 1
    dataset_dir = util.config_get_dataset_dir(config)
    dataset_dir_split = os.path.join(dataset_dir, split)
    dataset = image_dataset_from_directory(
        directory=dataset_dir_split,
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",  # means that the labels are encoded as integers
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config["image_size"],
        shuffle=shuffle,
        seed=1989
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128, seed=1989, reshuffle_each_iteration=True)

    return dataset

