"""TextCnnRnn model and associated helper functions."""
from typing import Union, TextIO, Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn

from incubator.glove import load_glove
from incubator.data import Vocabulary
from incubator.models.base_model import BaseModel


class LinearCnnRnn(BaseModel):
    """
    TextCnnRnn model from sequence of tokens to a class output.

    Structure:
        TokenIds -> Embedding -> Conv2d -> RNN -> Dense -> Output
    """

    def __init__(self,
                 embedding: nn.Embedding,
                 rnn: nn.RNNBase,
                 num_classes: int,
                 filters: List[int],
                 out_channels: int,
                 ):
        """Inintialises model with embedding, RNN and Conv2D parameters."""
        super().__init__()

        self.embedding = embedding
        self.rnn = rnn

        convs = []
        for filter_size in filters:
            assert filter_size % 2 != 0, "Filters must be odd."
            padding = filter_size // 2  # to preserve the height of the input

            conv = nn.Conv2d(
                in_channels=1,  # just the word embedding as a channel
                out_channels=out_channels,
                kernel_size=(filter_size, self.embedding.embedding_dim),
                padding=(padding, 0),
            )
            convs.append(conv)

        self.convs = nn.ModuleList(convs)

        self.output = nn.Linear(
            in_features=rnn.hidden_size,
            out_features=num_classes,
        )

    def forward(self,
                utterance_tokens: torch.Tensor,
                mask: torch.Tensor,
                label: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate model output.

        utterance_tokens: (batch, seq_len, vocab_len)
        mask: (batch,)
        label: (batch)
        """
        # width_in = embedding_dim
        # height_in = seq_len

        # (batch, height_in, width_in)
        embeddings = self.embedding(utterance_tokens)
        # (batch, in_channels=1, height_in, width_in)
        embeddings = embeddings.unsqueeze(1)
        # [(batch, out_channels, height_out, width_out=1)] * len(filters)
        conv_outs = [torch.relu(conv(embeddings)) for conv in self.convs]
        # [(batch, out_channels, height_out)] * len(filters)
        conv_outs = [conv.squeeze(3) for conv in conv_outs]
        # (batch, out_channels * len(filters), height_out)
        concat = torch.cat(conv_outs, dim=1)
        # (batch, height_out, out_channels * len(filters)) // see [1]
        rnn_in = concat.transpose(1, 2)
        # (batch, height_out, hidden_size)
        rnn_out, _ = self.rnn(rnn_in)
        # (batch, hidden_size)
        utterance = rnn_out[:, -1, :]
        # (batch, num_classes)
        output = self.output(utterance)

        return output, self.loss(output, mask, label)

        # [1] As height_in = seq_len, we're using height_out as the
        # 'sequence length' for the RNN
        #
        # [2] We need to use the functional variant of MaxPool1d because the
        # kernel_size will vary. This happens because we want to cover the
        # entire height of the convolutional output, and has to be done
        # dynamically.


def glove_linear_cnn_lstm(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        filters: List[int],
        out_channels: int,
        rnn_hidden_size: int = 100,
        rnn_num_layers: int = 1,
        bidirectional: bool = False,
        rnn_dropout: float = 0,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        ) -> LinearCnnRnn:
    """Return LinearCnnRnn with embedding layer initialised with GloVe."""
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)

    lstm = nn.LSTM(
        input_size=out_channels * len(filters),
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        bidirectional=bidirectional,
        dropout=rnn_dropout,
        batch_first=True,
    )

    return LinearCnnRnn(
        embedding=embedding,
        num_classes=num_classes,
        rnn=lstm,
        filters=filters,
        out_channels=out_channels,
    )
