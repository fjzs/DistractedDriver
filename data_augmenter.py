import albumentations as A
import numpy as np
import tensorflow as tf
from functools import partial

import matplotlib.pyplot as plt
import CONSTANTS
# Source: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/tensorflow-example.ipynb
# Source: https://github.com/albumentations-team/albumentations/issues/905


def apply_augmentation_to_batch(images_batch: tf.Tensor, img_height:int, img_width:int):
    """

    :param images_batch: shape (m, h, w, c)
    :param img_height:
    :param img_width:
    :return:
    """
    print("\naugmentation_function called")
    print(f"image has shape: {images_batch.shape}")
    print(f"image has type: {type(images_batch)}")

    transforms = A.Compose(transforms=[
        A.RandomCrop(height=380, width=540, p=1),
        A.Resize(height=img_height, width=img_width)
    ])

    augmented_images = []
    for img in images_batch:  # apply augmentation pipeline to single image (not to the batch)
        aug_data = transforms(image=img.astype('uint8'))
        augmented_images.append(aug_data['image'])
    return np.stack(augmented_images)

### CHECK
def process_data(images_batch, labels, height:int=480, width:int=640):
    """

    :param images: shape (m, h, w, c)
    :param labels: shape (m,)
    :param height:
    :param width:
    :return:
    """
    print("\nprocess_data called")
    print(f"image has shape: {images_batch.shape}")
    print(f"image has type: {type(images_batch)}")
    print(f"label has shape: {labels.shape}")
    print(f"label has type: {type(labels)}")
    inputs = tf.numpy_function(func=apply_augmentation_to_batch, inp=[images_batch, height, width], Tout=tf.uint8)
    return inputs, labels


def add_augmentations(dataset: tf.data.Dataset, img_height: int, img_width: int) -> tf.data.Dataset:
    # dataset_augmented = dataset.map(partial(process_data, img_height=img_height, img_width=img_width))
    dataset_augmented = dataset.map(process_data)
    # num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset_augmented

def set_shapes(img, label, img_shape=(480,640,3)):
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label


def unit_test_map(dataset: tf.data.Dataset):
    dataset_augmented = dataset.map(process_data)
    #dataset_augmented = dataset_augmented.map(set_shapes)
    images, labels = next(iter(dataset_augmented))  # extract 1 batch from the dataset

    plt.figure(figsize=(6.4*2, 4.8*3))
    num = 9
    for i in range(num):
        ax = plt.subplot(3, 3, i+1)
        img = images[i].numpy().astype("uint8")
        ax.imshow(img)
        label_index = int(labels[i])
        label = CONSTANTS.CLASSES[label_index]
        ax.set_title(label, {"fontsize":9})
    plt.show()


def unit_test(dataset: tf.data.Dataset):
    images, labels = next(iter(dataset))  # extract 1 batch from the dataset
    plt.figure(figsize=(3, 6))
    num = 4
    for i in range(num):
        idx1 = 2*i + 1
        ax = plt.subplot(num, 2, idx1)
        img = images[i].numpy().astype("uint8")
        ax.imshow(img)
        label_index = int(labels[i])
        label = CONSTANTS.CLASSES[label_index]
        plt.title(f"{label}", fontsize=9)

        #transformed img
        idx2 = 2 * i + 2
        ax = plt.subplot(num, 2, idx2)
        img_i = images[i]
        label_i = labels[i]
        aug_img, aug_label = process_data(img_i, label_i, 480, 640)
        t_img = aug_img.numpy().astype("uint8")
        ax.imshow(t_img)
        plt.title(f"Processed", fontsize=9)
    plt.show()
