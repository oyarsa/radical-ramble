from io import StringIO
from pathlib import Path

from incubator import glove, data

glove_str = (
"""test 0.1 0.2 0.3
new 0.4 0.5 0.6
"""
)

def test_synthetic_glove() -> None:
    "Test if GloVe loader works with synthetic data"
    vocab = data.Vocabulary([["test", "new"]])
    input_file = StringIO(glove_str)
    tensor = glove.load_glove(input_file, vocab, 3)

    assert tensor.shape == (vocab.vocab_size(), 3)


def test_real_glove() -> None:
    "Test if GloVe loader works with real data"
    glove_dim = 50

    data_path = Path('./data/dev/metadata.csv')
    dataset = data.MeldTextDataset(data_path)

    input_file = Path(f'./data/glove/glove.6B.{glove_dim}d.txt')
    tensor = glove.load_glove(input_file, dataset.vocab, glove_dim)

    assert tensor.shape == (dataset.vocab.vocab_size(), glove_dim)
