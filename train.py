import CONSTANTS
import tensorflow as tf

# Training Configurations
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
SEED = 42
VAL_FRACTION = 0.2


def load_dataset() -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        CONSTANTS.DIR_TRAIN_IMAGES,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_FRACTION,
        subset="training",
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    x=1




if __name__ == "__main__":
    dataset = load_dataset()
    y=2