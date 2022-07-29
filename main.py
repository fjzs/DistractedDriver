import dataset_generator
import train
import CONSTANTS


if __name__ == "__main__":
    # To generate the train/val folders
    # dataset_generator.generate_from_original_labels(total_fraction=0.01, val_fraction=0.3)

    # To train a model with a specific dataset
    config = {
        "model_name": "test1",
        "dataset": "data_001",
        "is_new_experiment": False,
        "image_size": (256, 256),
        "batch_size": 60,
        "epochs": 50
    }
    train.train_experiment(config)



