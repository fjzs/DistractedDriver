import keras.models

import util
from CONSTANTS import DIR_EXPERIMENTS, NUM_CLASSES
import tensorflow as tf
import dataset_loader
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from keras.applications import EfficientNetB2
from keras import models, layers, optimizers
from util import plot_and_save_logs
import os


def train_experiment(config: dict) -> None:

    print(f"Starting experiment with configuration:{config}")
    experiment_name = config["model_name"] + "_" + config["dataset"]
    is_new_experiment = config["is_new_experiment"]

    # Create the directory of the experiment if not existant
    folder_path = os.path.join(DIR_EXPERIMENTS, experiment_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Save the config file
    with open(os.path.join(folder_path, "config.txt"),"w") as file:
        file.write(str(config))

    # Assemble the train and load generators
    train_dataset, val_dataset = dataset_loader.load(config)

    # Callbacks
    csv_logger = get_callback_CSVLogger(folder_path)
    model_checkpoint = get_callback_ModelCheckpoint(folder_path)

    # Define the model or load it if its necessary
    model = None
    if is_new_experiment:
        model = create_model(config)
        model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])
    else:
        model = keras.models.load_model(os.path.join(folder_path,"best.hdf5"))

    # Cache the logs if this is a re-training
    df_previous = None
    if not is_new_experiment:
        df_previous = util.load_csv_logs(folder_path)

    # Train the model now
    model.fit(
        x=train_dataset,
        epochs=config["epochs"],
        verbose=1,
        callbacks=[csv_logger, model_checkpoint],
        validation_data=val_dataset
    )

    # Update the previous logs
    util.update_csv_logs(df_previous, folder_path)

    # Generate the plots of the logs
    plot_and_save_logs(folder_path)


def get_callback_CSVLogger(folder_path: str) -> CSVLogger:
    return CSVLogger(os.path.join(folder_path, 'logs.csv'))


def get_callback_ModelCheckpoint(folder_path: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        filepath=os.path.join(folder_path, "best.hdf5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min"
    )


def create_model(config: dict) -> models.Model:

    # Define the input shape
    input_shape = (config["image_size"][0], config["image_size"][1], 3)

    # Create the base model
    base_model = EfficientNetB2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        include_top=False,  # Do not include the ImageNet classifier at the top.
        pooling="max",
        input_shape=input_shape
    )
    base_model.trainable = False

    # Design the model now
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(units=NUM_CLASSES, activation="softmax"))
    print(model.summary())
    return model


if __name__ == "__main__":
    pass








