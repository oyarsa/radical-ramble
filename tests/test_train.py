"""Tests for incubator.train module."""
from typing import Any
from io import StringIO
import wandb
import torch

import incubator.datasets.meld_linear_text_dataset as mltd
import incubator.datasets.meld_contextual_text_dataset as mctd
from incubator.models.simple import glove_simple
from incubator.models.contextual_simple import glove_contextual_simple
from incubator.util import set_seed
from incubator.train import train, calc_accuracy

import tests.test_models as tm
from tests.helpers import read_test_data


def _noop(arg: Any) -> None:
    return


def test_linear_train(monkeypatch: Any) -> None:
    """Test training loop for a linear module."""
    set_seed(1000)

    monkeypatch.setattr(wandb, 'log', _noop)

    df = read_test_data()
    train_dataset = mltd.MeldLinearTextDataset(df, mode='emotion')
    dev_dataset = mltd.MeldLinearTextDataset(df, mode='emotion',
                                             vocab=train_dataset.vocab)
    glove_file = StringIO(tm.glove_str)

    train_loader = mltd.meld_linear_text_daloader(
        dataset=train_dataset,
        batch_size=tm.batch_size,
    )
    dev_loader = mltd.meld_linear_text_daloader(
        dataset=dev_dataset,
        batch_size=tm.batch_size,
    )

    classifier = glove_simple(
        glove_path=glove_file,
        glove_dim=tm.glove_dim,
        num_classes=tm.num_classes,
        vocab=train_dataset.vocab,
    )

    train(model=classifier, trainloader=train_loader, devloader=dev_loader)


def test_contextual_train(monkeypatch: Any) -> None:
    """Test training loop for a contextual module."""
    set_seed(1000)

    monkeypatch.setattr(wandb, 'log', _noop)

    df = read_test_data()
    train_dataset = mctd.MeldContextualTextDataset(df, mode='emotion')
    dev_dataset = mctd.MeldContextualTextDataset(df, mode='emotion',
                                                 vocab=train_dataset.vocab)
    glove_file = StringIO(tm.glove_str)

    train_loader = mctd.meld_contextual_text_daloader(
        dataset=train_dataset,
        batch_size=tm.batch_size,
    )
    dev_loader = mctd.meld_contextual_text_daloader(
        dataset=dev_dataset,
        batch_size=tm.batch_size,
    )

    classifier = glove_contextual_simple(
        glove_path=glove_file,
        glove_dim=tm.glove_dim,
        num_classes=tm.num_classes,
        vocab=train_dataset.vocab,
    )

    train(model=classifier, trainloader=train_loader, devloader=dev_loader)


def test_linear_accuracy() -> None:
    """Test accuracy calculation for linear model."""
    # Predictions will be [2, 0, 1]
    outputs = torch.tensor([
        [1, 3, 4],
        [3, 2, 2],
        [1, 3, 1],
    ])
    labels = torch.tensor([2, 1, 1])
    mask = torch.tensor([1, 1, 1])

    assert calc_accuracy(outputs, labels, mask) == 2


def test_contextual_accuracy() -> None:
    """Test accuracy calculation for contextual model."""
    # Predictions will be [[2, 0], [1, PAD]]
    outputs = torch.tensor([
        [
            [1, 3, 4],
            [3, 2, 2],
        ],
        [
            [1, 3, 1],
            [0, 0, 0],
        ],
    ])
    labels = torch.tensor([[2, 1], [1, 0]])  # last one is padded
    mask = torch.tensor([[1, 1], [1, 0]])

    assert calc_accuracy(outputs, labels, mask) == 2
