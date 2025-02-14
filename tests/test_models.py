"""Test model training with synthetic data."""
from io import StringIO

from incubator import data
from incubator.models.simple import (
    random_emb_simple,
    glove_simple,
)
from incubator.models.linear_rnn import glove_linear_lstm
from incubator.models.linear_cnn import glove_linear_cnn
from incubator.models.linear_cnn_rnn import glove_linear_cnn_lstm
from incubator.models.contextual_simple import glove_contextual_simple
from incubator.models.contextual_rnn import glove_contextual_lstm
from incubator.models.bc_lstm import glove_bc_lstm
from incubator.datasets.meld_linear_text_dataset import (
    MeldLinearTextDataset,
    meld_linear_text_daloader,
)
from incubator.datasets.meld_contextual_text_dataset import (
    MeldContextualTextDataset,
    meld_contextual_text_daloader,
)
from tests.helpers import read_test_data

batch_size = 3
embedding_dim = 50
num_classes = len(data.emotions)
glove_str = """oh 0.5 0.3 0.1
my 0.4 0.3 0.5
god 0.1 0.1 0.1
's 0.1 0.1 0.2
"""
glove_dim = 3


def test_random_simple() -> None:
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = random_emb_simple(
        vocab_size=dataset.vocab_size(),
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    for batch in loader:
        predictions, _ = classifier(batch.tokens, batch.labels)
        assert predictions.shape == (batch_size, num_classes)


def test_glove_simple() -> None:
    "Test if GloVe loader works with synthetic data"
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_simple(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
    )

    for batch in loader:
        predictions, _ = classifier(batch.tokens, batch.labels)
        assert predictions.shape == (batch_size, num_classes)


def test_linear_rnn() -> None:
    "Test if Linear Rnn GloVe loader works with synthetic data"
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_linear_lstm(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
    )

    for batch in loader:
        predictions, _ = classifier(batch.tokens, batch.labels)
        assert predictions.shape == (batch_size, num_classes)


def test_linear_cnn() -> None:
    "Test if Linear Cnn GloVe loader works with synthetic data"
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_linear_cnn(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
        filters=[2, 3, 4],
        out_channels=3,
    )

    for batch in loader:
        predictions, _ = classifier(batch.tokens, batch.labels)
        assert predictions.shape == (batch_size, num_classes)


def test_linear_cnn_rnn() -> None:
    """Test if Linear Cnn+Rnn model works with synthetic data."""
    df = read_test_data()
    dataset = MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=3,
    )

    classifier = glove_linear_cnn_lstm(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
        filters=[3, 5],
        out_channels=3,
    )

    for batch in loader:
        predictions, _ = classifier(batch.tokens, batch.labels)
        assert predictions.shape == (batch_size, num_classes)


def test_contextual_simple() -> None:
    """Test ContextualSimple model with synthetic data."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    batch_size = 2
    nutterances = 2

    loader = meld_contextual_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_contextual_simple(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
    )

    for batch in loader:
        predictions, loss = classifier(batch.tokens, batch.masks, batch.labels)
        assert predictions.shape == (batch_size, nutterances, num_classes)
        assert loss.shape == ()  # scalar


def test_contextual_rnn() -> None:
    """Test ContextualRnn model with synthetic data."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    batch_size = 2
    nutterances = 2

    loader = meld_contextual_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_contextual_lstm(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
    )

    for batch in loader:
        predictions, loss = classifier(batch.tokens, batch.masks, batch.labels)
        assert predictions.shape == (batch_size, nutterances, num_classes)
        assert loss.shape == ()  # scalar


def test_bc_lstm() -> None:
    """Test if bcLSTM works with synthetic data."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotion')
    glove_file = StringIO(glove_str)

    batch_size = 2
    nutterances = 2

    loader = meld_contextual_text_daloader(
        dataset=dataset,
        batch_size=batch_size,
    )

    classifier = glove_bc_lstm(
        glove_path=glove_file,
        glove_dim=glove_dim,
        num_classes=num_classes,
        vocab=dataset.vocab,
        filters=[3, 5],
        out_channels=3,
    )

    for batch in loader:
        predictions, loss = classifier(batch.tokens, batch.masks, batch.labels)
        assert predictions.shape == (batch_size, nutterances, num_classes)
        assert loss.shape == ()  # scalar
