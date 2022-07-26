import dataset_generator as dg
import dataset_loader as dl
import CONSTANTS



if __name__ == "__main__":
    # To generate csv datasets
    #dg.generate_from_original_labels(dev_fraction=0.2, directory=CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL)
    #dg.sample_from_created("data\\datasetv00.csv", 0.1, "datasetv01")

    # To train a model
    config = {
        "target_size": (256, 256),
        "batch_size": 16
    }



