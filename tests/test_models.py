from torch.utils.data.dataloader import DataLoader

from incubator import data
from incubator.models import SimpleClassifier
from tests.helpers import read_test_data, test_tokens

batch_size = 3
embedding_dim = 50
num_classes = len(data.emotions)

def test_simple_classifier() -> None:
    df = read_test_data()
    dataset = data.MeldTextDataset(df, mode='emotion')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data.padding_collate_fn
    )


    classifier = SimpleClassifier(
        vocab_size=dataset.vocab_size(),
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    for batch in loader:
        predictions = classifier(batch.utteranceTokens)
        assert predictions.shape == (batch_size, num_classes)

