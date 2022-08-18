import albumentations as A
import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
import CONSTANTS
# Source: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/tensorflow-example.ipynb
# Source: https://github.com/albumentations-team/albumentations/issues/905


def add_augmentations(dataset: tf.data.Dataset, config_augmentation: dict) -> tf.data.Dataset:
    """
    Add the augmentations specified in the config file and returns the transformed dataset
    :param dataset: a tf.data.Dataset object
    :param config_augmentation: the transformations and their params
    :return: dataset_augmented
    """
    if config_augmentation is not None and len(config_augmentation) > 0:
        augmentation_pipeline = create_ComposeObject(config_augmentation)
        dataset_augmented = dataset.map(partial(augment_images, augmentation_pipeline=augmentation_pipeline))
        return dataset_augmented
    else:
        return dataset


def create_ComposeObject(config_augmentation: dict) -> A.Compose:
    """
    Creates a Compose object from Albumentations library to do the transformations
    :param config_augmentation:
    :return: a Compose object
    """
    transforms = []
    for transform, params in config_augmentation.items():
        if transform == "RandomCrop":
            transforms.append(A.RandomCrop(height=params["height"],
                                           width=params["width"],
                                           p=params["p"]))
        elif transform == "Resize":
            transforms.append(A.Resize(height=params["height"],
                                       width=params["width"]))
        elif transform == "HorizontalFlip":
            transforms.append(A.HorizontalFlip(p=params["p"]))
        else:
            raise ValueError(f"transform {transform} not implemented")

    return A.Compose(transforms=transforms)


def augment_images(inputs, labels, augmentation_pipeline: A.Compose):
    """
    Applies the augmentation pipeline to a batch of images
    :param inputs:
    :param labels:
    :param augmentation_pipeline:
    :return:
    """

    def apply_augmentation(images):
        augmented_images = []
        for i in range(images.shape[0]):  # apply augmentation pipeline to single image (not to the batch)
            aug_data = augmentation_pipeline(image=images[i].astype('uint8'))
            augmented_images.append(aug_data['image'])
        return np.stack(augmented_images)

    inputs = tf.numpy_function(func=apply_augmentation, inp=[inputs], Tout=tf.uint8)
    return inputs, labels


def set_shapes(img, label, img_shape=(480,640,3)):
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label



def unit_test_map(dataset: tf.data.Dataset):
    dataset_augmented = dataset.map(augment_images)
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
        aug_img, aug_label = augment_images(img_i, label_i, 480, 640)
        t_img = aug_img.numpy().astype("uint8")
        ax.imshow(t_img)
        plt.title(f"Processed", fontsize=9)
    plt.show()
