import dataset_generator
from CONSTANTS import DIR_DATA, DIR_EXPERIMENTS
from analysis import evaluate_and_report
import train

if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.generate_from_original_labels(total_fraction=0.10, val_fraction=0.3)

    # To train a model with a specific dataset
    config_train = {
        "model_name": "01_noAug",
        "dataset": "data_100",
        "is_new_experiment": False,
        "image_size": (480, 640),  # height x width
        "batch_size": 10,
        "epochs": 50
    }
    config_augmentation = {
        #"HorizontalFlip": {"p":0.5}
    }
    #train.train_experiment(config_train, config_augmentation)

    print("\nEvaluating experiment...")
    evaluate_and_report(config_train)

    # TODO: add type checking with Pytype (https://theaisummer.com/best-practices-deep-learning-code/)
    # TODO: code style checker for TF: Pylint



