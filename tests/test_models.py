from io import StringIO
from torch.utils.data.dataloader import DataLoader

from incubator import data, glove
from incubator.models.simple_classifier import (
    random_emb_simple_classifier,
    glove_simple_classifier,
)
from incubator.datasets.meld_linear_text_dataset import (
    MeldLinearTextDataset,
    meld_linear_text_daloader,
)
from tests.helpers import read_test_data, test_tokens

batch_size = 3
embedding_dim = 50
num_classes = len(data.emotions)
glove_str = """oh 0.5 0.3 0.1
my 0.4 0.3 0.5
god 0.1 0.1 0.1
's 0.1 0.1 0.2
"""
glove_dim = 3

def test_simple_classifier() -> None:
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = random_emb_simple_classifier(
        vocab_size=dataset.vocab_size(),
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    for batch in loader:
        predictions = classifier(batch.utterance_tokens)
        assert predictions.shape == (batch_size, num_classes)


def test_glove_classifier() -> None:
    "Test if GloVe loader works with synthetic data"
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_simple_classifier(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
    )

    for batch in loader:
        predictions = classifier(batch.utterance_tokens)
        assert predictions.shape == (batch_size, num_classes)