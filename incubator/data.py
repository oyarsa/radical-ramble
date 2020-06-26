from pathlib import Path
from typing import Tuple, NamedTuple, List, TypeVar, Callable, Union, Dict, Set

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
    word_types: Set[str],
    pad_token: str,
    unk_token: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {pad_token: 0, unk_token: 1}
    idx2word = {0: pad_token, 1: unk_token}

    for i, word in enumerate(word_types, 2):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word

def get_word_types(texts: List[List[str]]) -> Set[str]:
    return set(word for text in texts for word in text)

class Vocabulary:
    def __init__(
        self,
        texts: List[List[str]],
        pad_token: str='<PAD>',
        unk_token: str='<UNK>',
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        word_types = get_word_types(texts)
        self._word2index, self._index2word = build_indexes(
            word_types, pad_token, unk_token)

    def word2index(self, word: str) -> int:
        if word in self._word2index:
            return self._word2index[word]
        else:
            return self._word2index[self.unk_token]

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
        return len(self._word2index)



def one_hot(index: int, length: int) -> torch.Tensor:
    tensor = torch.zeros(length)
    tensor[index] = 1
    return tensor


class DatasetRow(NamedTuple):
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

class DatasetBatch(NamedTuple):
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

class MeldTextDataset(Dataset): # type: ignore
    def __init__(
        self,
        raw_data: Union[pd.DataFrame, Path],
        mode: str ='sentiment',
    ):
        if isinstance(raw_data, Path):
            raw_data = pd.read_csv(raw_data)
        self.data = preprocess_data(raw_data)
        self.vocab = Vocabulary(list(self.data.Tokens))
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> DatasetRow:
        row = self.data.iloc[index]
        tokenIds = self.vocab.map_tokens_to_ids(row.Tokens)

        if self.mode == 'sentiment':
            label = one_hot(sentiment2index[row.Sentiment], len(sentiments))
        else:
            label = one_hot(emotion2index[row.Emotion], len(emotions))

        return DatasetRow(
            dialogueId=row.Dialogue_ID,
            utteranceId=row.Utterance_ID,
            tokens=torch.tensor(tokenIds),
            label=label,
        )

    def vocab_size(self) -> int:
        return self.vocab.vocab_size()


def padding_collate_fn(batch: List[DatasetRow]) -> DatasetBatch:
    sortedBatch = sorted(batch, key=lambda row: -len(row.tokens))
    lengths = torch.tensor([len(item.tokens) for item in sortedBatch])

    labels = torch.stack([item.label for item in sortedBatch], dim=0)
    utteranceIds = torch.tensor([item.utteranceId for item in sortedBatch])
    dialogueIds = torch.tensor([item.dialogueId for item in sortedBatch])

    tokensList = [item.tokens for item in sortedBatch]
    utteranceTokens = pad_sequence(tokensList, batch_first=True)

    return DatasetBatch(
        lengths=lengths,
        utteranceIds=utteranceIds,
        dialogueIds=dialogueIds,
        utteranceTokens=utteranceTokens,
        labels=labels
    )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    return chain_func(data,
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


T = TypeVar('T')
def chain_func(initialArg: T, *functions: Callable[[T], T]) -> T:
    result = initialArg
    for function in functions:
        result = function(result)
    return result
