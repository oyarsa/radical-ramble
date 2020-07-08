import os
from io import StringIO
from pathlib import Path
import pytest

from incubator import glove, data
from incubator.datasets.meld_linear_text_dataset import MeldLinearTextDataset

glove_str = ("test 0.1 0.2 0.3\n"
             "new 0.4 0.5 0.6")


def test_synthetic_glove() -> None:
    """Test if GloVe loader works with synthetic data."""
    vocab = data.Vocabulary(["test", "new"])
    input_file = StringIO(glove_str)
    tensor = glove.load_glove(input_file, vocab, 3)

    assert tensor.shape == (vocab.vocab_size(), 3)


@pytest.mark.skipif(os.environ.get('TEST', '') != 'ALL', reason="very slow")
def test_real_glove() -> None:
    """Test if GloVe loader works with real data."""
    glove_dim = 50

    data_path = Path('./data/dev/metadata.csv')
    dataset = MeldLinearTextDataset(data_path)

    input_file = Path(f'./data/glove/glove.6B.{glove_dim}d.txt')
    tensor = glove.load_glove(input_file, dataset.vocab, glove_dim)

    assert tensor.shape == (dataset.vocab.vocab_size(), glove_dim)
