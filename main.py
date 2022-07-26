import dataset_generator as dg
import CONSTANTS



if __name__ == "__main__":
    dg.generate_from_original_labels(dev_fraction=0.1, directory=CONSTANTS.DIR_IMAGES_ORIGINAL_LABEL)