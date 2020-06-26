from typing import TypeVar, Callable, Union, Iterable, List
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


T = TypeVar('T')
def chain_func(initialArg: T, *functions: Callable[[T], T]) -> T:
    result = initialArg
    for function in functions:
        result = function(result)
    return result


def flatten2list(object: Iterable[T]) -> List[T]:
    """
        Taken from https://symbiosisacademy.org/tutorial-index/python-flatten-nested-lists-tuples-sets/
    """
    gather = []
    for item in object:
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatten2list(item))
        else:
            gather.append(item)
    return gather
