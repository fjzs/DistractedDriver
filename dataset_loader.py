import CONSTANTS
import pandas as pd
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# How to: https://vijayabhaskar96.medium.com/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1

def load(config:dict):

    # Useful parameters from config
    dataset_path = os.path.join(CONSTANTS.DIR_DATA,config["dataset"])

    # Load the .csv dataframe
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found")
    df = pd.read_csv(dataset_path)
    print(f"Successfully loaded dataset {dataset_path}")

    # Display basic info of the splits
    display_basic_info(df)

    # Create the generators
    train_gen = create_generator(df.loc[df['split'] == "train"], "train")
    val_gen = create_generator(df.loc[df['val'] == "val"], "val")

    # Show some images
    for i, x in enumerate(train_gen):
        if i == 20:
            break



    return train_gen, val_gen


def create_generator(df: pd.DataFrame, split:str, config:dict):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        xcol="filepath",
        ycol="class",
        weight_col=None,
        target_size=config["target_size"],
        color_mode="rgb",
        classes=None,
        class_mode="sparse",
        batch_size=config["batch_size"],
        shuffle=True,
        seed=1989,
        save_to_dir=None,
        save_prefix=None,
        save_format=None,
        subset=None,
        interpolation='nearest',
        validate_filenames=False
    )
    return generator



def display_basic_info(df: pd.DataFrame):
    splits = ["train", "dev", "test"]
    classes = np.unique(df["class"])
    for split in splits:
        print(f"Split: {split}")
        number_of_images = (df["split"] == split).sum()
        print(f"\t# of images: {number_of_images}")
        for c in classes:
            number_of_images_class = ((df["split"] == split) & (df["class"] == c)).sum()
            print(f"\t\tclass {c}: {number_of_images_class}")


if __name__ == "__main__":
    config = {
        "target_size":(256,256),
        "batch_size": 16
    }
    load("data\\datasetv01.csv", config)