import dataset_generator
from CONSTANTS import DIR_DATA, DIR_EXPERIMENTS
from error_analysis import evaluate_and_report
import train

if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.generate_from_original_labels(total_fraction=0.10, val_fraction=0.3)

    # To train a model with a specific dataset
    config = {
        "model_name": "test_aug",
        "dataset": "data_test",
        "is_new_experiment": True,
        "image_size": (480, 640),  # height x width
        "batch_size": 9,
        "epochs": 5
    }
    train.train_experiment(config)

    #print("\nEvaluating experiment...")
    #evaluate_and_report(config)




