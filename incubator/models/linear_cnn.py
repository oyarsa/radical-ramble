"TextCnn model and associated helper functions"
from typing import Union, TextIO, Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from incubator.glove import load_glove
from incubator.data import Vocabulary
from incubator.models.base_model import BaseModel


class LinearCnn(BaseModel):
    """
    TextCnn model from sequence of tokens to a class output

        TokenIds -> Embedding -> Conv2d -> MaxPool1d -> Dense -> Output
    """
    def __init__(self,
                 embedding: nn.Embedding,
                 num_classes: int,
                 filters: List[int],
                 out_channels: int,
                 dropout: float = 0,
                 ):
        super(LinearCnn, self).__init__()

        self.embedding = embedding

        convs = []
        for filter_size in filters:
            conv = nn.Conv2d(
                in_channels=1,  # just the word embedding as a channel
                out_channels=out_channels,
                kernel_size=(filter_size, self.embedding.embedding_dim),
                padding=(filter_size - 1, 0),
            )
            convs.append(conv)

        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            in_features=len(filters) * out_channels,
            out_features=num_classes,
        )

    def forward(self, utteranceTokens: torch.Tensor,
                label: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        utteranceTokens: (batch, seq_len, vocab_len)
        """
        # width_in = embedding_dim
        # height_in = seq_len

        # (batch, height_in, width_in)
        embeddings = self.embedding(utteranceTokens)
        # (batch, in_channels=1, height_in, width_in)
        embeddings = embeddings.unsqueeze(1)
        # [(batch, out_channels, height_out, width_out=1)] * len(filters)
        conv_outs = [torch.relu(conv(embeddings)) for conv in self.convs]
        # [(batch, out_channels, height_out)] * len(filters)
        conv_outs = [conv.squeeze(3) for conv in conv_outs]
        # [(batch, out_channels, pooled_out=1)] * len(filters) // see [1]
        pooled = [F.max_pool1d(c, c.shape[2]) for c in conv_outs]
        # [(batch, out_channels)] * len(filters)
        pooled = [x.squeeze(2) for x in pooled]
        # (batch, len(filters) * out_channels)
        utterance = torch.cat(pooled, dim=1)
        utterance = self.dropout(utterance)
        # (batch, num_classes)
        output = self.output(utterance)

        return output, self.loss(output, label)

        # [1] We need to use the functional variant of MaxPool1d because the
        # kernel_size will vary. This happens because we want to cover the
        # entire height of the convolutional output, and has to be done
        # dynamically.


def random_emb_linear_cnn(
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        filters: List[int],
        out_channels: int,
        dropout: float = 0,
        ) -> LinearCnn:
    "LinearCnn with randomly initialised embeddings layer"
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    return LinearCnn(
        embedding=embedding,
        num_classes=num_classes,
        filters=filters,
        out_channels=out_channels,
        dropout=dropout,
    )


def glove_linear_cnn(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        filters: List[int],
        out_channels: int,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        dropout: float = 0,
        ) -> LinearCnn:
    "LinearCnn with embedding layer initialised with GloVe"
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)

    return LinearCnn(
        embedding=embedding,
        num_classes=num_classes,
        filters=filters,
        out_channels=out_channels,
        dropout=dropout,
    )
