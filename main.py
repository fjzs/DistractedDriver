import dataset_generator
from CONSTANTS import DIR_DATA, DIR_EXPERIMENTS
from analysis import evaluate_and_report
import train

if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.generate_from_original_labels(total_fraction=0.10, val_fraction=0.3)

    # To train a model with a specific dataset
    config_train = {
        "model_name": "test",
        "dataset": "data_001",
        "is_new_experiment": True,
        "image_size": (480, 640),  # height x width
        "batch_size": 16,
        "epochs": 5,
        "base_model_layers_to_fine_tune": 5
    }
    config_augmentation = {
        #"HorizontalFlip": {"p":0.5}
        "RandomCrop": {"height": 380, "width": 540, "p":0.5},
        "Resize": {"height": 480, "width": 640}
    }
    train.train_experiment(config_train, config_augmentation)

    print("\nEvaluating experiment...")
    evaluate_and_report(config_train)

    # TODO: add type checking with Pytype (https://theaisummer.com/best-practices-deep-learning-code/)
    # TODO: code style checker for TF: Pylint



