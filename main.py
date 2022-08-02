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
        "model_name": "test02_noAug",
        "dataset": "data_100",
        "is_new_experiment": True,
        "image_size": (640, 480),
        "batch_size": 8,
        "epochs": 5,
        "shuffle_dataset": True
    }
    #train.train_experiment(config)

    #print("\nEvaluating experiment...")
    #evaluate_and_report(config)




