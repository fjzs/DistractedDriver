import CONSTANTS
import tensorflow as tf
import dataset_loader
from tensorflow.keras.callbacks import CSVLogger
import os


def train_experiment(config: dict):

    print(f"Starting experiment with configuration:{config}")
    model_name = config["model_name"]
    experiment_name = model_name

    # Create the directory of the experiment
    folder_path = os.path.join(CONSTANTS.DIR_EXPERIMENTS, experiment_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Save the config file
    with open(os.path.join(folder_path, "config.txt"),"w") as file:
        file.write(str(config))

    # Assemble the train and load generators
    train_gen, val_gen = dataset_loader.load(config)

    # This will automatically log model performance to this file
    csv_logger = CSVLogger(os.path.join(folder_path, 'logs.log'))










