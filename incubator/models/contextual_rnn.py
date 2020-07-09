"""ContextualRnn model and associated helper functions."""
from typing import Union, TextIO, Optional, Tuple
from pathlib import Path
import torch.nn as nn
from torch import Tensor

from incubator.glove import load_glove
from incubator.data import Vocabulary
from incubator.models.base_model import BaseModel


class ContextualRnn(BaseModel):
    """
    Simple RNN-based model from sequence of tokens to a class output.

    Structure:
        TokenIds -> Embedding -> RNN -> Dense -> Output

    Where RNN is a standard GRU/LSTM without attention.
    """

    def __init__(self,
                 embedding: nn.Embedding,
                 utterance_rnn: nn.RNNBase,
                 dialogue_rnn: nn.RNNBase,
                 num_classes: int,
                 dropout: float = 0,
                 ):
        """Initialise model with embedding and RNN."""
        super().__init__()

        self.embedding = embedding
        self.utterance_rnn = utterance_rnn
        self.dialogue_rnn = dialogue_rnn
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            in_features=self.dialogue_rnn.hidden_size,
            out_features=num_classes,
        )

    # pylint: disable=arguments-differ
    def forward(self,
                dialogue_tokens: Tensor,
                mask: Tensor,
                label: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Calculate model output.

        utterance_tokens: (batch, nutterances, seq_len)
        mask: (batch,)
        label: (batch,)
        """
        batch, nutterances, seq_len = dialogue_tokens.shape
        # (batch, nutterances, seq_len, embedding_dim)
        embeddings = self.embedding(dialogue_tokens)
        embeddings = self.dropout(embeddings)
        # (batch * nutterances, seq_len, embedding_dim)
        embeddings = embeddings.reshape(batch * nutterances, seq_len, -1)
        # (batch * nutterances, seq_len, hidden_dim)
        utt_rnn_out, _ = self.utterance_rnn(embeddings)
        # (batch * nutterances, utt_hidden_dim)
        dialogue_embs = utt_rnn_out[:, -1, :]
        # (batch, nutterances, utt_hidden_dim)
        dialogue_embs = dialogue_embs.reshape(batch, nutterances, -1)
        # (batch, nutterances, dia_hidden_dim)
        dia_rnn_out, _ = self.dialogue_rnn(dialogue_embs)
        dia_rnn_out = self.dropout(dia_rnn_out)
        # (batch, n_utterances, num_classes)
        output = self.output(dia_rnn_out)

        return output, self.loss(output, mask, label)


def glove_contextual_lstm(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        rnn_hidden_size: int = 100,
        rnn_num_layers: int = 1,
        bidirectional: bool = False,
        rnn_dropout: float = 0,
        ) -> ContextualRnn:
    """
    Return ContextualRnn with embedding layer initialised with GloVe.

    Both RNNs used (utterance and dialogue) are LSTMs.
    """
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)
    utterance_lstm = nn.LSTM(
        input_size=embedding.embedding_dim,
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        bidirectional=bidirectional,
        dropout=rnn_dropout,
        batch_first=True,
    )
    dialogue_lstm = nn.LSTM(
        input_size=utterance_lstm.hidden_size,
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        bidirectional=False,  # this doesn't make sense to be bidirectional [1]
        dropout=rnn_dropout,
        batch_first=True,
    )

    return ContextualRnn(
        embedding=embedding,
        num_classes=num_classes,
        dialogue_rnn=dialogue_lstm,
        utterance_rnn=utterance_lstm,
    )

    #
    # [1]: Dialogue has to be analysed in a single direction. It doesn't make
    # any sense whatsoever to understand it backwards. For each utterance,
    # though, it does.
