import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader

from incubator import data
from tests.helpers import read_test_data, test_tokens

def test_clean_unicode() -> None:
    data_path = Path('./data/dev/metadata.csv')
    dev = pd.read_csv(data_path)
    dev = data.clean_unicode(dev)

    nonascii = dev[dev.Utterance.apply(lambda s: not s.isascii())]
    assert len(nonascii) == 0

def test_preprocessing() -> None:
    df = read_test_data()
    processed = data.preprocess_data(df)

    assert list(processed.iloc[0].Tokens) == test_tokens[0]
    assert list(processed.iloc[1].Tokens) == test_tokens[1]
    assert list(processed.iloc[2].Tokens) == test_tokens[2]

def test_word_types() -> None:
    df = read_test_data()
    processed = data.preprocess_data(df)

    word_types = data.get_word_types(list(df.Tokens))

    for tokens in test_tokens:
        assert all(token in word_types for token in tokens)

def test_build_indexes() -> None:
    "Test if every token has an index and the two-way mapping is right"
    df = read_test_data()
    processed = data.preprocess_data(df)
    word_types = data.get_word_types(list(df.Tokens))

    word2idx, idx2word = data.build_indexes(word_types)

    for sentence in test_tokens:
        for token in sentence:
            assert token in word2idx
            index = word2idx[token]
            assert idx2word[index] == token


def test_index_uniqueness() -> None:
    "Test if every token has an unique index"
    df = read_test_data()
    processed = data.preprocess_data(df)
    word_types = data.get_word_types(list(df.Tokens))

    word2idx, idx2word = data.build_indexes(word_types)

    for sentence in test_tokens:
        indexes = [word2idx[token] for token in set(sentence)]
        assert len(indexes) == len(set(indexes))


def test_dataset_emotion() -> None:
    df = read_test_data()
    dataset = data.MeldTextDataset(df, mode='emotion')

    assert dataset[0].dialogueId == 0
    assert dataset[0].utteranceId == 0
    assert dataset[0].label.equal(torch.tensor([0, 0, 0, 0, 0, 0, 1]).float())
    assert len(dataset[0].tokens) == len(test_tokens[0])

    assert dataset[1].dialogueId == 0
    assert dataset[1].utteranceId == 1
    assert dataset[1].label.equal(torch.tensor([0, 0, 0, 0, 0, 1, 0]).float())
    assert len(dataset[1].tokens) == len(test_tokens[1])

def test_dataset_sentiment() -> None:
    df = read_test_data()
    dataset = data.MeldTextDataset(df, mode='sentiment')

    assert dataset[0].dialogueId == 0
    assert dataset[0].utteranceId == 0
    assert dataset[0].label.equal(torch.tensor([0, 0, 1]).float())
    assert len(dataset[0].tokens) == len(test_tokens[0])

    assert dataset[1].dialogueId == 0
    assert dataset[1].utteranceId == 1
    assert dataset[1].label.equal(torch.tensor([0, 0, 1]).float())
    assert len(dataset[1].tokens) == len(test_tokens[1])

def test_dataloader() -> None:
    df = read_test_data()
    dataset = data.MeldTextDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=3,
        collate_fn=data.padding_collate_fn
    )
    length0 = len(test_tokens[0])
    length1 = len(test_tokens[1])
    length2 = len(test_tokens[2])
    max_length = max(length0, length1, length2)

    for batch in loader:
        assert batch.dialogueIds.equal(torch.tensor([1, 0, 0]))
        assert batch.utteranceIds.equal(torch.tensor([0, 0, 1]))
        assert batch.labels.equal(torch.tensor([[0, 1, 0],
                                                [0, 0, 1],
                                                [0, 0, 1]]).float())
        assert batch.lengths.equal(torch.tensor([length2, length0, length1]))
        assert all(len(seq) == max_length for seq in batch.utteranceTokens)

