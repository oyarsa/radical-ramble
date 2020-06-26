import os
import random
import torch
import numpy as np

def set_seed(seed: int) -> None:
    """
    This sets the seeds to everything, basically. Torch, CUDA, NumPy, stdlib
    random. Should be called in the beginning of every executable file.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # type: ignore
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
