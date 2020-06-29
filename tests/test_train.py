from io import StringIO

import incubator.datasets.meld_linear_text_dataset as mltd
from incubator.models.simple_classifier import glove_simple_classifier
from incubator.util import set_seed
from incubator.train import train

import tests.test_models as tm
from tests.helpers import read_test_data

def test_train() -> None:
    set_seed(1000)

    df = read_test_data()
    dataset = mltd.MeldLinearTextDataset(df, mode='emotion')
    glove_file = StringIO(tm.glove_str)

    loader = mltd.meld_linear_text_daloader(
        dataset=dataset,
        batch_size=tm.batch_size,
    )

    classifier = glove_simple_classifier(
        glove_path=glove_file,
        glove_dim=tm.glove_dim,
        num_classes=tm.num_classes,
        vocab=dataset.vocab,
    )

    train(model=classifier, trainloader=loader, devloader=loader)
