import dataset_generator
import train
import CONSTANTS


if __name__ == "__main__":
    # To generate csv datasets
    #dataset_generator.generate_from_original_labels(dev_fraction=0.2, directory=CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL)
    #dataset_generator.sample_from_created("data\\datasetv00.csv", 0.1, "datasetv01")

    # To train a model with a specific dataset
    config = {
        "model_name": "test1",
        "dataset": "datasetv01.csv",
        "target_size": (256, 256),
        "batch_size": 16
    }
    print(f"Starting experiment with configuration:{config}")
    train.train_experiment(config)



