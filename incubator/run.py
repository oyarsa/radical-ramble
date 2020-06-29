"CLI script for running the module."
import torch

from incubator import util
util.set_seed(1000)

if __name__ == '__main__':
    print(torch.cuda.is_available())
