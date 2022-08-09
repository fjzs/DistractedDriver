import data_augmenter
from CONSTANTS import DIR_EXPERIMENTS, NUM_CLASSES
from data_augmenter import add_augmentations_2
import dataset_loader
from keras.applications import EfficientNetB2
from keras import models, layers, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import os
import util


def train_experiment(config_train: dict, config_augmentation: dict) -> None:

    print(f"Starting experiment with configuration:{config_train}")
    is_new_experiment = config_train["is_new_experiment"]
    experiment_dir = util.config_get_experiment_dir(config_train)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Save the config file
    with open(os.path.join(experiment_dir, "config_train.txt"),"w") as file:
        file.write(str(config_train))

    # Assemble the train and val dataset
    train_dataset = dataset_loader.load_dataset_split("train", config_train, True)
    img_width, img_height = config_train["image_size"]
    #data_augmenter.unit_test_map(train_dataset)
    train_dataset = add_augmentations_2(train_dataset, config_augmentation)
    util.visualize_dataset(train_dataset)
    val_dataset = dataset_loader.load_dataset_split("val", config_train, True)

    # Callbacks
    csv_logger = get_callback_CSVLogger(experiment_dir)
    model_checkpoint = get_callback_ModelCheckpoint(experiment_dir)

    # Define the model or load it if its necessary
    model = None
    if is_new_experiment:
        model = create_model(config_train)
        model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])
    else:
        model = keras.models.load_model(os.path.join(experiment_dir,"best.hdf5"))

    # Cache the logs if this is a re-training
    df_previous = None
    if not is_new_experiment:
        df_previous = util.load_csv_logs(experiment_dir)

    # Train the model now
    model.fit(
        x=train_dataset,
        epochs=config_train["epochs"],
        verbose=1,
        callbacks=[csv_logger, model_checkpoint],
        validation_data=val_dataset
    )

    # Update the previous logs
    util.update_csv_logs(df_previous, experiment_dir)

    # Generate the plots of the logs
    util.plot_and_save_logs(experiment_dir)


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


if __name__ == "__main__":
    pass








