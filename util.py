import shutil
import os
import numpy as np


def fix_random_seed(seed):
    np.random.seed(seed)

# Fixing the random seed
fix_random_seed(4321)


def build_train_dev_dataset(train_dir:str, dev_dir:str, train_fraction:float = 0.1):

    # Train subfolders are the initial classes
    train_subfolders = [f.path for f in os.scandir(train_dir) if f.is_dir()]

    # If the subfolders in dev are not created, create them
    for subfolder in train_subfolders:
        dev_subfolder_dir = os.path.join(dev_dir, subfolder)
        if not os.path.isdir(dev_subfolder_dir):
            os.mkdir(dev_subfolder_dir)

    # For each subfolder, copy



