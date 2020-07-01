"RnnClassifier model and associated helper functions"
from typing import Union, TextIO, Optional
from pathlib import Path
import torch.nn as nn
from torch import Tensor

from incubator.glove import load_glove
from incubator.data import Vocabulary

class LinearRnn(nn.Module): # type: ignore
    """
    Simple RNN-based model from sequence of tokens to a class output

        TokenIds -> Embedding -> RNN -> Dense -> Output

    Where RNN is a standard GRU/LSTM without attention.
    """
    def __init__(self,
                 embedding: nn.Embedding,
                 rnn: nn.RNNBase,
                 num_classes: int):
        super(LinearRnn, self).__init__()

        self.embedding = embedding
        self.rnn = rnn

        self.output = nn.Linear(
            in_features=self.rnn.hidden_size,
            out_features=num_classes,
        )

    # pylint: disable=arguments-differ
    def forward(self, utteranceTokens: Tensor) -> Tensor:
        """
        utteranceTokens: (batch, seq_len, vocab_len)
        """
        # (batch, seq_len, embedding_dim)
        embeddings = self.embedding(utteranceTokens)
        # (batch, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(embeddings)
        # (batch, hidden_dim)
        utterance = rnn_out[:, -1, :]
        # (batch, num_classes)
        output = self.output(utterance)

        return output


def glove_linear_lstm(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        rnn_hidden_size: int = 100,
        rnn_num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
    ) -> LinearRnn:
    """
    RnnClassifier with embedding layer initialised with GloVe.
    RNN used is an LSTM.
    """
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)
    lstm = nn.LSTM(
        input_size=embedding.embedding_dim,
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        batch_first=True,
    )

    return LinearRnn(
        embedding=embedding,
        num_classes=num_classes,
        rnn=lstm,
    )
