import dataset_generator
from analysis import evaluate_and_report
import train

if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.generate_from_csv(fraction=0.05)

    # To train a model with a specific dataset
    config_train = {
        "model_name": "02_Resnet",
        "dataset": "data_noleak_005",
        "is_new_experiment": True,
        "image_size": (480, 640),  # height x width
        "batch_size": 32,
        "epochs": 50,
        "base_model_last_layers_to_fine_tune": 5,
        "model_type": "ResNet50"
    }
    config_augmentation = {
        #"HorizontalFlip": {"p":0.5}
        "RandomCrop": {"height": 380, "width": 540, "p":0.5},  # maintain
        "Resize": {"height": 480, "width": 640},  # maintain
        #"RandomBrightnessContrast": {"p": 0.5}
    }
    train.train_experiment(config_train, config_augmentation)
    evaluate_and_report(config_train)

    # TODO: add type checking with Pytype (https://theaisummer.com/best-practices-deep-learning-code/)
    # TODO: code style checker for TF: Pylint



