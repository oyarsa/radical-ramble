import os
from io import StringIO
from pathlib import Path

import pandas as pd
import torch

from incubator import data

test_data_str = """
Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
1,"Oh my God, hes lost it. Hes totally lost it.",Phoebe,sadness,negative,0,0,4,7,"00:20:57,256","00:21:00,049"
2,What?,Monica,surprise,negative,0,1,4,7,"00:21:01,927","00:21:03,261"
"""
test_tokens = [
    ['Oh', 'my', 'God', 'he', "'s", 'lost', 'it', 'He', "'s",
     'totally', 'lost', 'it'],
    ['What'],
]

def test_clean_unicode() -> None:
    data_path = Path('./data/dev/metadata.csv')
    dev = pd.read_csv(data_path)
    dev = data.clean_unicode(dev)

    nonascii = dev[dev.Utterance.apply(lambda s: not s.isascii())]
    assert len(nonascii) == 0

def test_preprocessing() -> None:
    raw_data = StringIO(test_data_str)
    df = pd.read_csv(raw_data)
    processed = data.preprocess_data(df)

    assert list(processed.iloc[0].Tokens) == test_tokens[0]
    assert list(processed.iloc[1].Tokens) == test_tokens[1]

def test_word_types() -> None:
    raw_data = StringIO(test_data_str)
    df = pd.read_csv(raw_data)
    processed = data.preprocess_data(df)

    word_types = data.get_word_types(list(df.Tokens))

    assert all(token in word_types for token in test_tokens[0])

def test_build_indexes() -> None:
    "Test if every token has an index and the two-way mapping is right"
    raw_data = StringIO(test_data_str)
    df = pd.read_csv(raw_data)
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
    raw_data = StringIO(test_data_str)
    df = pd.read_csv(raw_data)
    processed = data.preprocess_data(df)
    word_types = data.get_word_types(list(df.Tokens))

    word2idx, idx2word = data.build_indexes(word_types)

    for sentence in test_tokens:
        indexes = [word2idx[token] for token in set(sentence)]
        assert len(indexes) == len(set(indexes))


def test_dataset() -> None:
    raw_data = StringIO(test_data_str)
    df = pd.read_csv(raw_data)
    dataset = data.MeldTextDataset(df)

    assert dataset[0].dialogueId == 0
    assert dataset[0].utteranceId == 0
    assert dataset[0].emotionId == data.emotion2index['sadness']
    assert dataset[0].sentimentId == data.sentiment2index['negative']
    assert len(dataset[0].tokenIds) == len(test_tokens[0])

    assert dataset[1].dialogueId == 0
    assert dataset[1].utteranceId == 1
    assert dataset[1].emotionId == data.emotion2index['surprise']
    assert dataset[1].sentimentId == data.sentiment2index['negative']
    assert len(dataset[1].tokenIds) == len(test_tokens[1])