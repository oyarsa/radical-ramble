"Functions for manipulating the MELD dataset"
from typing import Tuple, List, Dict, Set

import pandas as pd
import spacy
import torch
from spacy.lang.en import English

from incubator.util import chain_func

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
    word2idx = {pad_token: 0, unk_token: 1}
    idx2word = {0: pad_token, 1: unk_token}

    for i, word in enumerate(word_types, 2):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word

def get_word_types(words: List[str]) -> Set[str]:
    return set(word for word in words)

class Vocabulary:
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
        if word in self._word2index:
            return self._word2index[word]
        return self._word2index[self.unk_token]

    def index2word(self, index: int) -> str:
        if index in self._index2word:
            return self._index2word[index]
        return self._index2word[0]

    def map_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word2index(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.index2word(id) for id in ids]

    def vocab_size(self) -> int:
        return len(self._word2index)



def one_hot(index: int, length: int) -> torch.Tensor:
    tensor = torch.zeros(length)
    tensor[index] = 1
    return tensor


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    return chain_func(
        data,
        lower_case,
        clean_unicode,
        tokenise,
        remove_punctuation,
    )


def lower_case(data: pd.DataFrame) -> pd.DataFrame:
    data['Utterance'] = data.Utterance.str.lower()
    return data


def clean_unicode(data: pd.DataFrame) -> pd.DataFrame:
    "Replace the Unicode characters with their appropriate replacements."
    data['Utterance'] = (
        data.Utterance.apply(lambda s: s.replace('\x92', "'"))
        .apply(lambda s: s.replace('\x85', ". "))
        .apply(lambda s: s.replace('\x97', " "))
        .apply(lambda s: s.replace('\x91', ""))
        .apply(lambda s: s.replace('\x93', ""))
        .apply(lambda s: s.replace('\xa0', ""))
        .apply(lambda s: s.replace('\x94', ""))
    )
    return data


def get_tokeniser() -> spacy.tokenizer.Tokenizer:
    """
    Create a Tokenizer with the default settings for English
    including punctuation rules and exceptions.
    """
    nlp = English()
    tokeniser = nlp.Defaults.create_tokenizer(nlp)
    return tokeniser


def tokenise(data: pd.DataFrame) -> pd.DataFrame:
    tokeniser = get_tokeniser()
    data['Tokens'] = data.Utterance.apply(tokeniser)
    return data


def remove_punctuation(data: pd.DataFrame) -> pd.DataFrame:
    def filter_punct(tokens: spacy.tokens.Doc) -> List[str]:
        return [token.text for token in tokens if not token.is_punct]
    data.Tokens = data.Tokens.apply(filter_punct)
    return data
