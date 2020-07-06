from typing import Any
from io import StringIO
import wandb

import incubator.datasets.meld_linear_text_dataset as mltd
from incubator.models.simple import glove_simple
from incubator.util import set_seed
from incubator.train import train

import tests.test_models as tm
from tests.helpers import read_test_data


def test_train(monkeypatch: Any) -> None:
    set_seed(1000)

    def noop(arg: Any) -> None:
        return

    monkeypatch.setattr(wandb, 'log', noop)

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
