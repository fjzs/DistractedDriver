import dataset_generator as dg
import CONSTANTS



if __name__ == "__main__":
    #dg.generate_from_original_labels(dev_fraction=0.2, directory=CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL)
    dg.sample_from_created("data\\datasetv00.csv", 0.1, "datasetv01")
