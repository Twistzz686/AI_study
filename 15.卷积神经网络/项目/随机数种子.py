import os
import numpy as np

import random

import torch

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available,Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available,Using CPU.")