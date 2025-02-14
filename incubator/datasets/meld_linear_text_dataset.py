"""MeldLinearTextDataset class and assorted helper functions."""
from typing import NamedTuple, List, Optional, cast
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

from incubator.data import (
    Vocabulary,
    preprocess_data,
    sentiment2index,
    emotion2index,
)
from incubator.util import DataFrameOrFilePath


class LinearTextDatasetRow(NamedTuple):
    """
    Tuple to hold the fields for a given row of data in tensor form.

    Represents a single row/instance.
    """

    dialogue_id: int
    utterance_id: int
    tokens: torch.Tensor
    label: torch.Tensor

    def __str__(self) -> str:  # pragma: no cover
        """Format output for debugging."""
        return (f'DatasetRow\n'
                f'  dialogue_id: {self.dialogue_id}\n'
                f'  utterance_id: {self.utterance_id}\n'
                f'  tokens: {self.tokens}\n'
                f'  label: {self.label}'
                )


class LinearTextDatasetBatch(NamedTuple):
    """
    Tuple to hold the fields for a given batch of data in tensor form.

    Represents a whole batch. Tensors are batch-first.
    """

    dialogue_ids: torch.Tensor
    tokens: torch.Tensor
    utterance_ids: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    masks: torch.Tensor

    def __str__(self) -> str:  # pragma: no cover
        """Format output for debugging."""
        return (f'DatasetBatch\n'
                f'  dialogue_ids: {self.dialogue_ids}\n'
                f'  utterance_ids: {self.utterance_ids}\n'
                f'  tokens:\n    {self.tokens}\n'
                f'  labels:\n    {self.labels}\n'
                f'  lengths: {self.lengths}\n'
                f'  masks: {self.masks}'
                )


class MeldLinearTextDataset(Dataset):
    """
    Dataset for simple, linear text.

    Utterances are considered separately, without considering dialogues.
    """

    data: pd.DataFrame
    vocab: Vocabulary
    mode: str

    def __init__(self,
                 data: DataFrameOrFilePath,
                 vocab: Optional[Vocabulary] = None,
                 mode: str = 'sentiment'):
        """Initialise Dataset with data, vocab and mode."""
        if isinstance(data, (Path, str)):
            data = pd.read_csv(data)
        self.data = preprocess_data(data)

        if vocab is None:
            vocab = Vocabulary.build_vocab(self.data)
        self.vocab = vocab

        self.mode = mode

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> LinearTextDatasetRow:
        """Return item from dataset based on index."""
        row = self.data.iloc[index]
        token_ids = self.vocab.map_tokens_to_ids(row['Tokens'])

        if self.mode == 'sentiment':
            sentiment = cast(str, row['Sentiment'])
            label = torch.tensor(sentiment2index[sentiment])
        else:
            emotion = cast(str, row['Emotion'])
            label = torch.tensor(emotion2index[emotion])

        return LinearTextDatasetRow(
            dialogue_id=cast(int, row['Dialogue_ID']),
            utterance_id=cast(int, row['Utterance_ID']),
            tokens=torch.tensor(token_ids),
            label=label,
        )

    def vocab_size(self) -> int:
        """Return the size of the vocabulary built from this data."""
        return self.vocab.vocab_size()


def _padding_collate_fn(
        batch: List[LinearTextDatasetRow]
        ) -> LinearTextDatasetBatch:
    sorted_batch = sorted(batch, key=lambda row: -len(row.tokens))
    lengths = torch.tensor([len(item.tokens) for item in sorted_batch])

    labels = torch.stack([item.label for item in sorted_batch], dim=0)
    utterance_ids = torch.tensor([item.utterance_id for item in sorted_batch])
    dialogue_ids = torch.tensor([item.dialogue_id for item in sorted_batch])

    tokens_list = [item.tokens for item in sorted_batch]
    utterance_tokens = pad_sequence(tokens_list, batch_first=True)
    masks = torch.ones(len(batch)).long()

    return LinearTextDatasetBatch(
        lengths=lengths,
        utterance_ids=utterance_ids,
        dialogue_ids=dialogue_ids,
        tokens=utterance_tokens,
        labels=labels,
        masks=masks,
    )


def meld_linear_text_daloader(
        dataset: MeldLinearTextDataset,
        batch_size: int
        ) -> DataLoader:
    """Return a DataLoader configured with its batching function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_padding_collate_fn,
    )
