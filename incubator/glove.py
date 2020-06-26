from typing import TextIO, Union
from pathlib import Path
import torch

from incubator.data import Vocabulary

TextFile = Union[Path, TextIO]

def load_glove(
    input_file: TextFile,
    vocab: Vocabulary,
    glove_dim: int,
) -> torch.Tensor:
    weight_matrix = torch.zeros(size=(vocab.vocab_size(), glove_dim))
    if isinstance(input_file, Path):
        input_file = open(input_file, 'r')

    for line in input_file:
        word, *weights_str = line.split()
        weights = [float(weight) for weight in weights_str]
        weight_matrix[vocab.word2index(word)] = torch.tensor(weights)

    input_file.close()

    return weight_matrix
