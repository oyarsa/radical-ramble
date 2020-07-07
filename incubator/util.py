"Utility functions that don't fit elsewhere."
from typing import TypeVar, Callable, Iterable, List, Union
import os
import random
import pathlib

import torch
import numpy as np
import pandas

DataFrameOrFilePath = Union[pandas.DataFrame, pathlib.Path, str]


def set_seed(seed: int) -> None:
    """
    This sets the seeds to everything, basically. Torch, CUDA, NumPy, stdlib
    random. Should be called in the beginning of every executable file.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    np.random.seed(seed)  # type: ignore
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


T = TypeVar('T')


def chain_func(initial_arg: T, *functions: Callable[[T], T]) -> T:
    """
    Chain-applies `functions`, starting with `initial_args` and using the
    output from a function as input to the next.
    """
    result = initial_arg
    for function in functions:
        result = function(result)
    return result


def flatten2list(iterable: Iterable[T]) -> List[T]:
    """
    Taken from https://symbiosisacademy.org/tutorial-index/python-flatten-nested-lists-tuples-sets/  # NOQA
    """
    gather = []
    for item in iterable:
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatten2list(item))
        else:
            gather.append(item)
    return gather
