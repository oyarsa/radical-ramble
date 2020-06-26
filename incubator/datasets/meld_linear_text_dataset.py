from typing import NamedTuple, Union, List
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from incubator.data import (
    Vocabulary,
    one_hot,
    preprocess_data,
    sentiment2index,
    sentiments,
    emotion2index,
    emotions
)
from incubator.util import flatten2list

class LinearTextDatasetRow(NamedTuple):
    dialogueId: int
    utteranceId: int
    tokens: torch.Tensor
    label: torch.Tensor

    def __str__(self) -> str:
        return (f'DatasetRow\n'
                f'  dialogueId: {self.dialogueId}\n'
                f'  utteranceId: {self.utteranceId}\n'
                f'  tokens: {self.tokens}\n'
                f'  label: {self.label}'
        )

class LinearTextDatasetBatch(NamedTuple):
    dialogueIds: torch.Tensor
    utteranceIds: torch.Tensor
    utteranceTokens: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor

    def __str__(self) -> str:
        return (f'DatasetBatch\n'
                f'  dialogueIds: {self.dialogueIds}\n'
                f'  utteranceIds: {self.utteranceIds}\n'
                f'  utteranceTokens:\n    {self.utteranceTokens}\n'
                f'  labels:\n    {self.labels}\n'
                f'  lengths: {self.lengths}'
        )

class MeldLinearTextDataset(Dataset): # type: ignore
    def __init__(
        self,
        raw_data: Union[pd.DataFrame, Path],
        mode: str ='sentiment',
    ):
        if isinstance(raw_data, Path):
            raw_data = pd.read_csv(raw_data)
        self.data = preprocess_data(raw_data)
        lst = list(self.data.Tokens)
        words = flatten2list(lst)
        self.vocab = Vocabulary(words)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> LinearTextDatasetRow:
        row = self.data.iloc[index]
        tokenIds = self.vocab.map_tokens_to_ids(row.Tokens)

        if self.mode == 'sentiment':
            label = one_hot(sentiment2index[row.Sentiment], len(sentiments))
        else:
            label = one_hot(emotion2index[row.Emotion], len(emotions))

        return LinearTextDatasetRow(
            dialogueId=row.Dialogue_ID,
            utteranceId=row.Utterance_ID,
            tokens=torch.tensor(tokenIds),
            label=label,
        )

    def vocab_size(self) -> int:
        return self.vocab.vocab_size()


def padding_collate_fn(batch: List[LinearTextDatasetRow]) -> LinearTextDatasetBatch:
    sortedBatch = sorted(batch, key=lambda row: -len(row.tokens))
    lengths = torch.tensor([len(item.tokens) for item in sortedBatch])

    labels = torch.stack([item.label for item in sortedBatch], dim=0)
    utteranceIds = torch.tensor([item.utteranceId for item in sortedBatch])
    dialogueIds = torch.tensor([item.dialogueId for item in sortedBatch])

    tokensList = [item.tokens for item in sortedBatch]
    utteranceTokens = pad_sequence(tokensList, batch_first=True)

    return LinearTextDatasetBatch(
        lengths=lengths,
        utteranceIds=utteranceIds,
        dialogueIds=dialogueIds,
        utteranceTokens=utteranceTokens,
        labels=labels
    )
