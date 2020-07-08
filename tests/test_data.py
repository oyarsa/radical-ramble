from incubator.data import preprocess_data
from pathlib import Path

import pandas as pd
import torch

from incubator import data
from incubator.util import flatten2list
from incubator.datasets.meld_linear_text_dataset import (
    MeldLinearTextDataset,
    meld_linear_text_daloader,
)
from incubator.datasets.meld_contextual_text_dataset import (
    MeldContextualTextDataset,
    meld_contextual_text_daloader,
)
from tests.helpers import read_test_data, test_tokens

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


def test_clean_unicode() -> None:
    data_path = Path('./data/dev/metadata.csv')
    dev = pd.read_csv(data_path)
    dev = data.clean_unicode(dev)

    nonascii = dev[dev['Utterance'].apply(lambda s: not s.isascii())]
    assert len(nonascii) == 0


def test_preprocessing() -> None:
    df = read_test_data()
    processed = data.preprocess_data(df)

    assert list(processed.iloc[0]['Tokens']) == test_tokens[0]
    assert list(processed.iloc[1]['Tokens']) == test_tokens[1]
    assert list(processed.iloc[2]['Tokens']) == test_tokens[2]


def test_word_types() -> None:
    df = read_test_data()
    df = preprocess_data(df)
    words = flatten2list(list(df['Tokens']))

    word_types = data.get_word_types(words)

    for tokens in test_tokens:
        assert all(token in word_types for token in tokens)


def test_build_indexes() -> None:
    "Test if every token has an index and the two-way mapping is right"
    df = read_test_data()
    df = preprocess_data(df)
    words = flatten2list(list(df['Tokens']))

    word_types = data.get_word_types(words)

    word2idx, idx2word = data.build_indexes(word_types, PAD_TOKEN, UNK_TOKEN)

    for sentence in test_tokens:
        for token in sentence:
            assert token in word2idx
            index = word2idx[token]
            assert idx2word[index] == token


def test_index_uniqueness() -> None:
    "Test if every token has an unique index"
    df = read_test_data()
    df = preprocess_data(df)
    words = flatten2list(list(df['Tokens']))

    word_types = data.get_word_types(words)

    word2idx, idx2word = data.build_indexes(word_types, PAD_TOKEN, UNK_TOKEN)

    for sentence in test_tokens:
        indexes = [word2idx[token] for token in set(sentence)]
        assert len(indexes) == len(set(indexes))


def test_linear_dataset_emotion() -> None:
    df = read_test_data()
    df = preprocess_data(df)
    dataset = MeldLinearTextDataset(df, mode='emotion')

    assert dataset[0].dialogue_id == 0
    assert dataset[0].utterance_id == 0
    assert dataset[0].label.equal(torch.tensor(6))
    assert len(dataset[0].tokens) == len(test_tokens[0])

    assert dataset[1].dialogue_id == 0
    assert dataset[1].utterance_id == 1
    assert dataset[1].label.equal(torch.tensor(5))
    assert len(dataset[1].tokens) == len(test_tokens[1])


def test_linear_dataset_sentiment() -> None:
    df = read_test_data()
    df = preprocess_data(df)
    dataset = MeldLinearTextDataset(df, mode='sentiment')

    assert dataset[0].dialogue_id == 0
    assert dataset[0].utterance_id == 0
    assert dataset[0].label.equal(torch.tensor(2))
    assert len(dataset[0].tokens) == len(test_tokens[0])

    assert dataset[1].dialogue_id == 0
    assert dataset[1].utterance_id == 1
    assert dataset[1].label.equal(torch.tensor(2))
    assert len(dataset[1].tokens) == len(test_tokens[1])


def test_linear_dataloader() -> None:
    df = read_test_data()
    dataset = MeldLinearTextDataset(df)
    loader = meld_linear_text_daloader(
        dataset=dataset,
        batch_size=3,
    )
    length0 = len(test_tokens[0])
    length1 = len(test_tokens[1])
    length2 = len(test_tokens[2])
    max_length = max(length0, length1, length2)

    for batch in loader:
        assert batch.dialogue_ids.equal(torch.tensor([1, 0, 0]))
        assert batch.utterance_ids.equal(torch.tensor([0, 0, 1]))
        assert batch.labels.equal(torch.tensor([1, 2, 2]))
        assert batch.lengths.equal(torch.tensor([length2, length0, length1]))
        assert all(len(seq) == max_length for seq in batch.tokens)


def test_contextual_dataset_emotion() -> None:
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotion')

    max_len0 = max(len(test_tokens[0]), len(test_tokens[1]))

    assert dataset[0].dialogue_id == 0
    assert dataset[0].labels.equal(torch.tensor([6, 5]))
    assert len(dataset[0].tokens) == 2
    assert len(dataset[0].tokens[0]) == max_len0
    assert len(dataset[0].tokens[1]) == max_len0

    assert dataset[1].dialogue_id == 1
    assert dataset[1].labels.equal(torch.tensor([4]))
    assert len(dataset[1].tokens) == 1
    assert len(dataset[1].tokens[0]) == len(test_tokens[2])


def test_contextual_dataset_sentiment() -> None:
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='sentiment')

    max_len0 = max(len(test_tokens[0]), len(test_tokens[1]))

    assert dataset[0].dialogue_id == 0
    assert dataset[0].labels.equal(torch.tensor([2, 2]))
    assert len(dataset[0].tokens) == 2
    assert len(dataset[0].tokens[0]) == max_len0
    assert len(dataset[0].tokens[1]) == max_len0

    assert dataset[1].dialogue_id == 1
    assert dataset[1].labels.equal(torch.tensor([1]))
    assert len(dataset[1].tokens) == 1
    assert len(dataset[1].tokens[0]) == len(test_tokens[2])


def test_contextual_dataloader() -> None:
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotions')
    loader = meld_contextual_text_daloader(
        dataset=dataset,
        batch_size=2,
    )
    length0 = len(test_tokens[0])
    length1 = len(test_tokens[1])
    length2 = len(test_tokens[2])
    max_length = max(length0, length1, length2)

    for batch in loader:
        assert batch.dialogue_ids.equal(torch.tensor([1, 0]))
        assert batch.labels.equal(torch.tensor([[4, 0], [6, 5]]))
        assert batch.lengths.equal(torch.tensor([[length2, 0],
                                                 [length0, length1]]))
        assert batch.tokens.shape == (2, 2, max_length)


def test_one_hot() -> None:
    """Test one_hot function for turning category into one-hot vector."""
    label = 3
    num_classes = 5
    vector = data.one_hot(label, num_classes)

    assert vector.equal(torch.tensor([0, 0, 0, 1, 0]).float())


def test_vocabulary_unknown_word() -> None:
    """Test if vocabulary returns unkown index for unknown word."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotions')
    vocab = dataset.vocab

    unkown_index = vocab.word2index(vocab.unk_token)
    assert vocab.word2index('COMPLETELYIMPOSSIBLETOKNOW') == unkown_index


def test_vocabulary_unknown_index() -> None:
    """Test if vocabulary returns unkown word for unknown index."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotions')
    vocab = dataset.vocab

    assert vocab.index2word(100000) == vocab.unk_token


def test_vocabulary_known_index() -> None:
    """Test if vocabulary returns correct word for known index."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotions')
    vocab = dataset.vocab

    word = vocab.index2word(2)  # first word non PAD or UNK for non-empty vocab
    assert word != vocab.unk_token


def test_vocabulary_token_id_map() -> None:
    """Test if vocab can correctly map list of words to integers and back."""
    df = read_test_data()
    dataset = MeldContextualTextDataset(df, mode='emotions')
    vocab = dataset.vocab

    words = ['oh', 'my', 'god']
    indexes = vocab.map_tokens_to_ids(words)
    new_words = vocab.map_ids_to_tokens(indexes)

    assert words == new_words


def test_vocabularies_are_equal() -> None:
    """Test if vocabularies by all datasets are equal to each other."""
    df = read_test_data()
    context_dataset = MeldContextualTextDataset(df, mode='sentiment')
    linear_dataset = MeldLinearTextDataset(df, mode='sentiment')

    assert context_dataset.vocab_size() == linear_dataset.vocab_size()
