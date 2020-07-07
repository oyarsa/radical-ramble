"""GloVe-related functions for loading."""
from typing import TextIO, Union, cast, Optional
from pathlib import Path
import torch

from incubator.data import Vocabulary

TextFile = Union[Path, str, TextIO]


def load_glove(
        input_file: TextFile,
        vocab: Vocabulary,
        glove_dim: int,
        saved_glove_file: Optional[Path] = None,
        ) -> torch.Tensor:
    """Load GloVe weight matrix based on a Vocabulary."""
    if saved_glove_file and saved_glove_file.exists():
        print('Loading GloVe file from', saved_glove_file)
        return cast(torch.Tensor, torch.load(saved_glove_file))

    weight_matrix = torch.zeros(size=(vocab.vocab_size(), glove_dim))
    if isinstance(input_file, (Path, str)):
        input_file = open(input_file, 'r')

    for line in input_file:
        word, *weights_str = line.split()
        weights = [float(weight) for weight in weights_str]
        weight_matrix[vocab.word2index(word)] = torch.tensor(weights)

    input_file.close()

    if saved_glove_file:
        print('Saving GloVe file to', saved_glove_file)
        torch.save(weight_matrix, saved_glove_file)

    return weight_matrix
