import os
import tensorflow as tf
import util
from keras.utils import image_dataset_from_directory
# Source: https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5


def load_dataset_split(dataset_name:str, split:str, image_size: tuple, batch_size:int, shuffle:bool, prefetch:bool=True) -> tf.data.Dataset:
    """
    loads a specific split of the data from a given directory and returns a tf.data.Dataset
    :param dataset_name:
    :param split:
    :param image_size: Tuple of (height, width)
    :param batch_size: use 1 if split is val or test
    :param shuffle:
    :param prefetch:
    :return:
    """

    if split not in ["train", "val", "test"]:
        raise ValueError(f"split parameter not recognized, it is: {split}")

    if split in ["val", "test"] and batch_size != 1:
        raise ValueError("batch_size must be 1 when split is val or test")

    if len(image_size) != 2:
        raise ValueError(f"image_size must be a tuple of 2 dimension, it is {image_size}")

    dataset_dir = util.config_get_dataset_dir(dataset_name)
    dataset_dir_split = os.path.join(dataset_dir, split)
    dataset = image_dataset_from_directory(
        directory=dataset_dir_split,
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",  # means that the labels are encoded as integers
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=util.SEED
    )

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=128, seed=util.SEED, reshuffle_each_iteration=True)

    return dataset

