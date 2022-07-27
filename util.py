import shutil
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import CONSTANTS


def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Fixing the random seed
fix_random_seed(1989)


def shuffle(input: list) -> None:
    random.shuffle(input)


def visualize_dataset(dataset: tf.data.Dataset) -> None:
    # Show some images
    plt.figure(figsize=(16, 16))
    for images, labels in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(CONSTANTS.CLASSES[int(labels[i])])
            plt.axis("off")
    plt.show()

