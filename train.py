import CONSTANTS
import tensorflow as tf
import dataset_loader
from tensorflow.keras.callbacks import CSVLogger
import os


def train_experiment(config: dict):

    model_name = config["model_name"]
    dataset = config["dataset"].split(".")[0]  # taking out the extension
    experiment_name = model_name + "_" + dataset

    # Create the directory of the experiment
    folder_path = os.path.join(CONSTANTS.DIR_MODELS, experiment_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Assemble the train and load generators
    train_gen, val_gen = dataset_loader.load()

    # This will automatically log model performance to this file
    csv_logger = CSVLogger(os.path.join(folder_path, 'logs.log'))










