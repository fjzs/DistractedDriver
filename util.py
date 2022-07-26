import shutil
import os
import numpy as np
import random

def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Fixing the random seed
fix_random_seed(1989)

def shuffle(input: list):
    random.shuffle(input)


