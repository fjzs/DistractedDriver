import shutil
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import CONSTANTS


def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Fixing the random seed
fix_random_seed(1989)


def shuffle(input: list) -> None:
    random.shuffle(input)


def load_csv_logs(folder_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(folder_path, "logs.csv"))


def update_csv_logs(df_initial: pd.DataFrame, folder_path: str) -> None:
    if df_initial is not None:
        last_epoch = df_initial["epoch"].max() + 1  # epochs start from zero
        df_additional = load_csv_logs(folder_path)
        df_additional["epoch"] = df_additional["epoch"] + last_epoch
        final_df = pd.concat([df_initial, df_additional])
        final_df.to_csv(os.path.join(folder_path, "logs.csv"), index=False)


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


def plot_and_save_logs(folder_path: str) -> None:
    """
    Plots the logs from the training history and saves them in a png file
    :param folder_path: path of the folder containing the logs.csv file
    :return: None
    """
    df = pd.read_csv(os.path.join(folder_path, "logs.csv"))
    metric_name_train = df.columns[1]
    metric_name_val = "val_" + metric_name_train
    fig, (ax_loss, ax_acc) = plt.subplots(1,2)
    fig.suptitle("Log history")
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # Losses plot
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epochs")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True)
    ax_loss.plot(df["epoch"], df["loss"], label="train loss", color="red")
    ax_loss.plot(df["epoch"], df["val_loss"], label="val loss", color="blue")
    ax_loss.legend(loc="upper right")

    # Losses plot
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epochs")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True)
    ax_acc.plot(df["epoch"], df[metric_name_train], label="train acc", color="red")
    ax_acc.plot(df["epoch"], df[metric_name_val], label="val acc", color="blue")
    ax_acc.legend(loc="upper right")
    plt.savefig(os.path.join(folder_path,"logs.png"))
    print(f"Successfully saved plot into {folder_path}")


if __name__ == "__main__":
    plot_and_save_logs("experiments\\test1_data_001")
