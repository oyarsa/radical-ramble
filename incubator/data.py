"Functions for manipulating the MELD dataset"
from typing import Tuple, List, Dict, Set, Iterable

import pandas as pd
import torch
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens.doc import Doc

from incubator.util import chain_func, flatten2list

sentiments = [
    'positive',
    'neutral',
    'negative'
]
sentiment2index = {
    sentiment: index for index, sentiment in enumerate(sentiments)
}

emotions = [
    'anger',
    'disgust',
    'fear',
    'joy',
    'neutral',
    'surprise',
    'sadness'
]
emotion2index = {
    emotion: index for index, emotion in enumerate(emotions)
}

def build_indexes(
        word_types: Set[str],
        pad_token: str,
        unk_token: str
        ) -> Tuple[Dict[str, int], Dict[int, str]]:
    "Builds word -> index and index -> word maps."
    word2idx = {pad_token: 0, unk_token: 1}
    idx2word = {0: pad_token, 1: unk_token}

    for i, word in enumerate(word_types, 2):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word


def get_word_types(words: List[str]) -> Set[str]:
    "Builds set of word types from list of words"
    return set(word for word in words)


class Vocabulary:
    "Holds the indexes for converting between tokens and token ids"
    pad_token: str
    unk_token: str
    _word2index: Dict[str, int]
    _index2word: Dict[int, str]

    def __init__(
            self,
            words: List[str],
            pad_token: str = '<PAD>',
            unk_token: str = '<UNK>',
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        word_types = get_word_types(words)
        self._word2index, self._index2word = build_indexes(
            word_types, pad_token, unk_token)

    def word2index(self, word: str) -> int:
        "Returns the id of `word`"
        if word in self._word2index:
            return self._word2index[word]
        return self._word2index[self.unk_token]

    def index2word(self, index: int) -> str:
        "Returns the word for `index`"
        if index in self._index2word:
            return self._index2word[index]
        return self._index2word[0]

    def map_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        "Maps a list of tokens to their list of ids"
        return [self.word2index(token) for token in tokens]

    def map_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        "Maps a list of ids back to the original tokens"
        return [self.index2word(id) for id in ids]

    def vocab_size(self) -> int:
        "Total size of the vocabulary, i.e., number of words"
        return len(self._word2index)

    @classmethod
    def build_vocab(cls, data: pd.DataFrame) -> 'Vocabulary':
        "Builds vocabulary from pre-processed data"
        words = flatten2list(data['Tokens'])
        return Vocabulary(words)


def one_hot(index: int, length: int) -> torch.Tensor:
    "Convert integer to one-hot vector representation"
    tensor = torch.zeros(length)
    tensor[index] = 1
    return tensor


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    "Chain-applies data pre-processing functions"
    return chain_func(
        data,
        lower_case,
        clean_unicode,
        tokenise,
        remove_punctuation,
    )


def lower_case(data: pd.DataFrame) -> pd.DataFrame:
    "Converts all strings to lower case"
    data['Utterance'] = data['Utterance'].str.lower()
    return data


def clean_unicode(data: pd.DataFrame) -> pd.DataFrame:
    "Replace the Unicode characters with their appropriate replacements."
    data['Utterance'] = (
        data['Utterance'].apply(lambda s: s.replace('\x92', "'"))
        .apply(lambda s: s.replace('\x85', ". "))
        .apply(lambda s: s.replace('\x97', " "))
        .apply(lambda s: s.replace('\x91', ""))
        .apply(lambda s: s.replace('\x93', ""))
        .apply(lambda s: s.replace('\xa0', ""))
        .apply(lambda s: s.replace('\x94', ""))
    )
    return data


def get_tokeniser() -> Tokenizer:
    """
    Create a Tokenizer with the default settings for English
    including punctuation rules and exceptions.
    """
    nlp = English()
    tokeniser = nlp.Defaults.create_tokenizer(nlp)
    return tokeniser


def tokenise(data: pd.DataFrame) -> pd.DataFrame:
    "Tokenises strings using the tokeniser from `get_tokeniser`"
    tokeniser = get_tokeniser()
    data['Tokens'] = data['Utterance'].apply(tokeniser)
    return data


def remove_punctuation(data: pd.DataFrame) -> pd.DataFrame:
    "Removes punctuation tokens"
    def filter_punct(tokens: Doc) -> List[str]:
        return [token.text for token in tokens if not token.is_punct]
    data['Tokens'] = data['Tokens'].apply(filter_punct)
    return data
