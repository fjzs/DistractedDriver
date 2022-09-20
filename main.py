import dataset_generator
from analysis import evaluate_and_report
import train

if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.sample_from_existing_dataset(fraction_train=0.01, fraction_val=0.01, fraction_test=0.1)

    # Configuration of model
    config_model = {
        "model_name": "02_finetune341_RC",
        "is_new": False,
        "image_size": (480, 640),  # height x width
        "batch_size": 2,
        "epochs": 30,
        "base_model_last_layers_to_fine_tune": 341,
        "model_type": "EfficientNetB2",
        "dropout_p": 0
    }

    # Configuration of data to use
    config_data = {
        "dataset": "data_noleak_005",
    }

    config_augmentation = {
        #"HorizontalFlip": {"p":0.5}
        "RandomCrop": {"height": 380, "width": 540, "p":0.5},  # maintain
        "Resize": {"height": 480, "width": 640},  # maintain
        #"RandomBrightnessContrast": {"p": 0.5}
        #"Cutout": {"num_holes": 10, "max_h_size": 60, "max_w_size": 60, "p": 0.3}
        #"Blur": {"blur_limit": 10, "p":0.3}
        #"Rotate": {"limit": 20, "p": 0.3}
    }
    #train.train_experiment(config_train, config_augmentation)
    evaluate_and_report(config_model, config_data, split="val")

    # TODO: add type checking with Pytype (https://theaisummer.com/best-practices-deep-learning-code/)
    # TODO: code style checker for TF: Pylint



