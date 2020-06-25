from pathlib import Path
from typing import Tuple, NamedTuple, List, TypeVar, Callable, Union, Dict, Set

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from spacy.lang.en import English

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
    word_types: Set[str]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {'UNK': 0}
    idx2word = {0: 'UNK'}

    for i, word in enumerate(word_types, 1):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word


def get_word_types(texts: List[List[str]]) -> Set[str]:
    return set(word for text in texts for word in text)


class Vocabulary:
    def __init__(self, texts: List[List[str]]):
        self._word_types = get_word_types(texts)
        self._word2index, self._index2word = build_indexes(self._word_types)

    def word2index(self, word: str) -> int:
        if word in self._word2index:
            return self._word2index[word]
        else:
            return self._word2index['UNK']

    def index2word(self, index: int) -> str:
        if index in self._index2word:
            return self._index2word[index]
        else:
            return self._index2word[0]

    def map_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word2index(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.index2word(id) for id in ids]

    def vocab_size(self) -> int:
        return len(self._word_types)


class DatasetRow(NamedTuple):
    dialogueId: int
    utteranceId: int
    tokenIds: torch.Tensor
    sentimentId: int
    emotionId: int

class MeldTextDataset(Dataset): # type: ignore
    def __init__(self, raw_data: Union[pd.DataFrame, Path]):
        if isinstance(raw_data, Path):
            raw_data = pd.read_csv(raw_data)
        self.data = preprocess_data(raw_data)
        self.token_vocab = Vocabulary(list(self.data.Tokens))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> DatasetRow:
        row = self.data.iloc[index]
        tokenIds = self.token_vocab.map_tokens_to_ids(row.Tokens)
        return DatasetRow(
            dialogueId=row.Dialogue_ID,
            utteranceId=row.Utterance_ID,
            tokenIds=torch.tensor(tokenIds),
            sentimentId=sentiment2index[row.Sentiment],
            emotionId=emotion2index[row.Emotion],
        )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    return chain_func(data,
        clean_unicode,
        tokenise,
        remove_punctuation,
    )


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


T = TypeVar('T')
def chain_func(initialArg: T, *functions: Callable[[T], T]) -> T:
    result = initialArg
    for function in functions:
        result = function(result)
    return result
