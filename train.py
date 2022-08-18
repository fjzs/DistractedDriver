import data_augmenter
from CONSTANTS import DIR_EXPERIMENTS, NUM_CLASSES
from data_augmenter import add_augmentations
import dataset_loader
from keras.applications import EfficientNetB2
from keras import models, layers, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
import os
import time
import util


def train_experiment(config_train: dict, config_augmentation: dict) -> None:

    print(f"Starting experiment with configuration:{config_train}")
    is_new_experiment = config_train["is_new_experiment"]
    experiment_dir = util.config_get_experiment_dir(config_train)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Save the config files
    with open(os.path.join(experiment_dir, "config_train.txt"),"w") as file:
        file.write(str(config_train))
    with open(os.path.join(experiment_dir, "config_augmentation.txt"),"w") as file:
        file.write(str(config_augmentation))

    # Assemble the train and val dataset
    train_dataset = dataset_loader.load_dataset_split("train", config_train, True)
    train_dataset = add_augmentations(train_dataset, config_augmentation)
    #util.visualize_dataset(train_dataset)
    val_dataset = dataset_loader.load_dataset_split("val", config_train, True)

    # Callbacks
    csv_logger = get_callback_CSVLogger(experiment_dir)
    model_checkpoint = get_callback_ModelCheckpoint(experiment_dir)

    # Define the model or load it if its necessary
    model = None
    if is_new_experiment:
        model = create_model2(config_train)
        model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])
    else:
        model = models.load_model(os.path.join(experiment_dir,"best.hdf5"))

    # Cache the logs if this is a re-training
    df_previous = None
    if not is_new_experiment:
        df_previous = util.load_csv_logs(experiment_dir)

    # Add the batching feature to the datasets
    train_dataset.batch(batch_size=config_train["batch_size"], num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset.batch(batch_size=config_train["batch_size"], num_parallel_calls=tf.data.AUTOTUNE)

    # Train the model now
    time_init = time.time()
    model.fit(
       x=train_dataset,
       epochs=config_train["epochs"],
       verbose=1,
       callbacks=[csv_logger, model_checkpoint],
       validation_data=val_dataset
    )
    time_elapsed = round(time.time() - time_init, 0)
    print(f"Elapsed seconds in training = {time_elapsed} seconds")
    sec_per_epoch = time_elapsed/config_train["epochs"]
    print(f"Avg seconds/epoch = {round(sec_per_epoch,1)}")

    # Update the previous logs
    util.update_csv_logs(df_previous, experiment_dir)

    # Generate the plots of the logs
    util.plot_and_save_logs(experiment_dir)


def get_callback_CSVLogger(folder_path: str) -> CSVLogger:
    return CSVLogger(os.path.join(folder_path, 'logs.csv'))


def get_callback_ModelCheckpoint(folder_path: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        filepath=os.path.join(folder_path, "best.hdf5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="max"
    )


def create_model(config: dict) -> models.Model:

    # Define the input shape
    input_shape = (config["image_size"][0], config["image_size"][1], 3)

    # Create the base model
    # Note that EfficientNetB2 already includes a preprocessing layer, it receives raw images
    base_model = EfficientNetB2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        include_top=False,  # Do not include the ImageNet classifier at the top.
        pooling="max",
        input_shape=input_shape
    )
    base_model.trainable = False
    print(base_model.summary())

    # Design the model now
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(units=NUM_CLASSES, activation="softmax"))
    print(model.summary())
    return model


def create_model2(config: dict) -> models.Model:

    # Define the input shape
    input_shape = (config["image_size"][0], config["image_size"][1], 3)

    # Create the base model
    # Note that EfficientNetB2 already includes a preprocessing layer, it receives raw images
    base_model = EfficientNetB2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        include_top=False,  # Do not include the ImageNet classifier at the top.
        pooling="max",
        input_shape=input_shape
    )
    base_model.trainable = False

    # Design the model with Functional API so it works with Grad CAM
    x = base_model.output
    pred_layer = layers.Dense(units=NUM_CLASSES, activation="softmax", name="prediction")(x)
    model = models.Model(inputs=base_model.input, outputs=pred_layer)
    print(model.summary())
    return model


if __name__ == "__main__":
    pass








