import dataset_generator
import train
import CONSTANTS


if __name__ == "__main__":
    # To generate the train/val folders
    #dataset_generator.generate_from_original_labels(val_fraction=0.2)

    # To train a model with a specific dataset
    config = {
        "model_name": "test1",
        "target_size": (256, 256),
        "batch_size": 16
    }
    train.train_experiment(config)



