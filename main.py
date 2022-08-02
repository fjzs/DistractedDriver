import dataset_generator
from CONSTANTS import DIR_DATA, DIR_EXPERIMENTS
from error_analysis import evaluate_and_report
import os
import train




if __name__ == "__main__":
    # To generate the train/val folders
    # dataset_generator.generate_from_original_labels(total_fraction=0.01, val_fraction=0.3)

    # To train a model with a specific dataset
    config = {
        "model_name": "test1",
        "dataset": "data_001",
        "is_new_experiment": False,
        "image_size": (256, 256),
        "batch_size": 64,
        "epochs": 100,
        "shuffle_dataset": True
    }
    train.train_experiment(config)
    evaluate_and_report(config)




