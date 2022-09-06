from CONSTANTS import DIR_EXPERIMENTS, NUM_CLASSES
from data_augmenter import add_augmentations
import dataset_loader
import keras
from keras.applications import EfficientNetB2, ResNet50
from keras import models, layers, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
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
    val_dataset = dataset_loader.load_dataset_split("val", config_train, True)

    # Activate this line to see the training examples with the augmentation pipeline
    util.visualize_dataset(train_dataset)

    # Callbacks
    csv_logger = get_callback_CSVLogger(experiment_dir)
    model_checkpoint = get_callback_ModelCheckpoint(experiment_dir)
    early_stopping = get_callback_EarlyStopping()

    # Define the model or load it if its necessary
    model = None
    if is_new_experiment:
        model = create_model(config_train)
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
       callbacks=[csv_logger, model_checkpoint, early_stopping],
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


def get_callback_EarlyStopping(monitor= "val_accuracy", min_delta=0.001, patience=10, verbose=1, mode="max") -> EarlyStopping:
    return EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode=mode,
        baseline=None,
        restore_best_weights=False
    )


def create_model(config: dict) -> models.Model:

    # Define the input shape
    input_shape = (config["image_size"][0], config["image_size"][1], 3)

    # Create the base model
    base_model = create_base_model(config["model_type"], input_shape, config["base_model_last_layers_to_fine_tune"])

    # https://stackoverflow.com/questions/70998847/transfer-learning-fine-tuning-how-to-keep-batchnormalization-in-inference-mode
    # Design the model with Functional API so it works with Grad CAM
    x = base_model.output

    # Check if dropout is added
    if config["dropout_p"] > 0:
        x = layers.Dropout(config["dropout_p"])(x)

    pred_layer = layers.Dense(units=NUM_CLASSES, activation="softmax", name="prediction")(x)
    model = models.Model(inputs=base_model.input, outputs=pred_layer)
    print(model.summary(expand_nested=True, show_trainable=True))
    return model


def create_base_model(model_type: str, input_shape, base_model_last_layers_to_fine_tune: int) -> keras.Model:
    base_model = None
    if model_type == "EfficientNetB2":
        base_model = EfficientNetB2(
            # Note that EfficientNetB2 already includes a preprocessing layer, it receives raw images
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            include_top=False,  # Do not include the ImageNet classifier at the top.
            pooling="max",
            input_shape=input_shape
        )
        # Freeze the first layers
        base_model.trainable = True
        total_num_layers = len(base_model.layers)
        for i in range(0, total_num_layers - 1 - base_model_last_layers_to_fine_tune):
            layer_i = base_model.get_layer(index=i)
            layer_i.trainable = False

    elif model_type == "ResNet50":  # ResNet50 requires a preprocessing layer initially
        base_model0 = ResNet50(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            include_top=False,  # Do not include the ImageNet classifier at the top.
            pooling="max",
            input_shape=input_shape
        )
        # Freeze the first layers
        base_model0.trainable = True
        total_num_layers = len(base_model0.layers)
        for i in range(0, total_num_layers - 1 - base_model_last_layers_to_fine_tune):
            layer_i = base_model0.get_layer(index=i)
            layer_i.trainable = False

        # Append the preprocessing layer
        inputs = keras.Input(shape=input_shape)
        x = layers.Lambda(keras.applications.resnet.preprocess_input, input_shape=input_shape, name="imagenet_preprocess")(inputs)
        model_with_preprocess = base_model0(x)
        base_model = models.Model(inputs=inputs, outputs=model_with_preprocess)
    else:
        raise ValueError(f"Model type {model_type} not implemented")

    return base_model







