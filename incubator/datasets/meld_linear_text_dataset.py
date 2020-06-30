from typing import NamedTuple, Union, List
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
from incubator.util import flatten2list

class LinearTextDatasetRow(NamedTuple):
    dialogue_id: int
    utterance_id: int
    tokens: torch.Tensor
    label: torch.Tensor

    def __str__(self) -> str:
        return (f'DatasetRow\n'
                f'  dialogue_id: {self.dialogue_id}\n'
                f'  utterance_id: {self.utterance_id}\n'
                f'  tokens: {self.tokens}\n'
                f'  label: {self.label}'
                )

class LinearTextDatasetBatch(NamedTuple):
    dialogue_ids: torch.Tensor
    utterance_tokens: torch.Tensor
    utterance_ids: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor

    def __str__(self) -> str:
        return (f'DatasetBatch\n'
                f'  dialogue_ids: {self.dialogue_ids}\n'
                f'  utterance_ids: {self.utterance_ids}\n'
                f'  utterance_tokens:\n    {self.utterance_tokens}\n'
                f'  labels:\n    {self.labels}\n'
                f'  lengths: {self.lengths}'
                )


class MeldLinearTextDataset(Dataset): # type: ignore
    def __init__(self,
                 data: Union[pd.DataFrame, Path],
                 mode: str = 'sentiment'):
        if isinstance(data, Path):
            data = pd.read_csv(data)
        self.data = preprocess_data(data)
        lst = list(self.data.Tokens)
        words = flatten2list(lst)
        self.vocab = Vocabulary(words)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> LinearTextDatasetRow:
        row = self.data.iloc[index]
        token_ids = self.vocab.map_tokens_to_ids(row.Tokens)

        if self.mode == 'sentiment':
            label = torch.tensor(sentiment2index[row.Sentiment])
        else:
            label = torch.tensor(emotion2index[row.Emotion])

        return LinearTextDatasetRow(
            dialogue_id=row.Dialogue_ID,
            utterance_id=row.Utterance_ID,
            tokens=torch.tensor(token_ids),
            label=label,
        )

    def vocab_size(self) -> int:
        return self.vocab.vocab_size()


def _padding_collate_fn(batch: List[LinearTextDatasetRow]) -> LinearTextDatasetBatch:
    sorted_batch = sorted(batch, key=lambda row: -len(row.tokens))
    lengths = torch.tensor([len(item.tokens) for item in sorted_batch])

    labels = torch.stack([item.label for item in sorted_batch], dim=0)
    utterance_ids = torch.tensor([item.utterance_id for item in sorted_batch])
    dialogue_ids = torch.tensor([item.dialogue_id for item in sorted_batch])

    tokens_list = [item.tokens for item in sorted_batch]
    utterance_tokens = pad_sequence(tokens_list, batch_first=True)

    return LinearTextDatasetBatch(
        lengths=lengths,
        utterance_ids=utterance_ids,
        dialogue_ids=dialogue_ids,
        utterance_tokens=utterance_tokens,
        labels=labels
    )

def meld_linear_text_daloader(
        dataset: MeldLinearTextDataset,
        batch_size: int
    ) -> DataLoader: # type: ignore
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_padding_collate_fn,
    )
