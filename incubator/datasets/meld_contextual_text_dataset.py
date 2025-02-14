"""MeldContextualTextDataset class and assorted helper functions."""
from typing import NamedTuple, List, Optional, cast, Tuple
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


class ContextualInstance(NamedTuple):
    """
    Stores the fields for a instance of the dataset.

    Each instance encodes a whole dialogue. The labels are the labels for
    each utterance in the dialogue, and the utteranceTokens represent the
    tokens for each utterance.

    In an dialogue with 4 utterances, each with max 10 tokens, we'll have:
    - labels: 4-d
    - utteranceTokens: 4x10-d
    """

    dialogue_id: int
    labels: torch.Tensor
    lengths: torch.Tensor
    tokens: torch.Tensor


class ContextualTextDatasetBatch(NamedTuple):
    """
    Stores the fields for a given batch of data in tensor form.

    Represents a whole batch. Tensors are batch-first.
    """

    dialogue_ids: torch.Tensor
    tokens: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor

    def __str__(self) -> str:  # pragma: no cover
        """Format output for debuggig."""
        return (f'DatasetBatch\n'
                f'  dialogue_ids: {self.dialogue_ids}\n'
                f'  tokens:\n    {self.tokens}\n'
                f'  labels:\n    {self.labels}\n'
                f'  lengths: {self.lengths}'
                )


class MeldContextualTextDataset(Dataset):
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
        """Initialise Dataset with the data, a vocab and mode."""
        if isinstance(data, (Path, str)):
            data = pd.read_csv(data)
        self.data = preprocess_data(data)

        if vocab is None:
            vocab = Vocabulary.build_vocab(self.data)
        self.vocab = vocab

        self.mode = mode

    def __len__(self) -> int:
        """Return length of the dataset."""
        return self.data['Dialogue_ID'].nunique()

    def __getitem__(self, index: int) -> ContextualInstance:
        """
        Return valid item from dataset based on index.

        It works by iterating through the indexes starting from `index` until
        it finds a valid item.
        """
        while index < len(self):
            item = self._item(index)
            if item is not None:
                return item
            index += 1
        raise IndexError

    def _item(self, index: int) -> Optional[ContextualInstance]:
        """
        Return item from dataset based on index, or None if index is invalid.

        Returns None if the item is invalid. In this case, it can happen if
        there is an empty Dialogue ID. For example, MELD's train.csv does not
        have rows for the Dialogue ID = 60.
        """
        rows = self.data[self.data['Dialogue_ID'] == index]

        utterances = []
        labels = []
        lengths = []

        for _, row in rows.iterrows():
            tokens = row['Tokens']
            if len(tokens) == 0:
                continue

            token_ids = self.vocab.map_tokens_to_ids(tokens)

            if self.mode == 'sentiment':
                sentiment = cast(str, row['Sentiment'])
                label = sentiment2index[sentiment]
            else:
                emotion = cast(str, row['Emotion'])
                label = emotion2index[emotion]

            utterances.append(torch.tensor(token_ids))
            labels.append(label)
            lengths.append(len(token_ids))

        if len(utterances) == 0:
            return None

        return ContextualInstance(
            dialogue_id=index,
            tokens=pad_sequence(utterances, batch_first=True),
            labels=torch.tensor(labels),
            lengths=torch.tensor(lengths),
        )

    def vocab_size(self) -> int:
        """Return the size of the vocabulary built from this data."""
        return self.vocab.vocab_size()


def _extend_sequence(tensor: torch.Tensor, new_size: Tuple[int, int],
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    max_seq_len, max_num_tokens = new_size
    seq_len, num_tokens = tensor.shape
    new_tensor = torch.zeros(max_seq_len, max_num_tokens).long()
    new_tensor[:seq_len, :num_tokens] = tensor

    mask = torch.ones(max_seq_len).long()
    mask[seq_len:].fill_(0)

    return new_tensor, mask


def _padding_collate_fn(
        batch: List[ContextualInstance]
        ) -> ContextualTextDatasetBatch:
    max_num_tokens = max(inst.lengths.max().item() for inst in batch)
    sorted_batch = sorted(batch, key=lambda inst: -inst.lengths.max().item())
    lengths = pad_sequence([item.lengths for item in sorted_batch],
                           batch_first=True)

    labels = pad_sequence([item.labels for item in sorted_batch],
                          batch_first=True)
    dialogue_ids = torch.tensor([item.dialogue_id for item in sorted_batch])

    tokens_list = [item.tokens for item in sorted_batch]

    max_length = max(item.shape[0] for item in tokens_list)
    new_size = (max_length, cast(int, max_num_tokens))

    extended_result = [_extend_sequence(utt, new_size) for utt in tokens_list]
    tokens_list, masks_list = list(zip(*extended_result))

    dialogue_tokens = torch.stack(tokens_list, dim=0)
    dialogue_masks = torch.stack(masks_list, dim=0)

    return ContextualTextDatasetBatch(
        lengths=lengths,
        tokens=dialogue_tokens,
        dialogue_ids=dialogue_ids,
        labels=labels,
        masks=dialogue_masks,
    )


def meld_contextual_text_daloader(
        dataset: MeldContextualTextDataset,
        batch_size: int
        ) -> DataLoader:
    """Return a DataLoader configured with its batching function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_padding_collate_fn,
    )
