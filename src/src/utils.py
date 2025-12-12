import numpy as np
import random

SEED = 42

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)