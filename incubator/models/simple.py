"""SimpleClassifier model and associated helper functions."""
from typing import Union, TextIO, Optional, Tuple
from pathlib import Path
import torch.nn as nn
from torch import Tensor

from incubator.glove import load_glove
from incubator.data import Vocabulary
from incubator.models.base_model import BaseModel


class Simple(BaseModel):
    """
    Simplest possible model from sequence of tokens to a class output.

    Structure:
        TokenIds -> Embedding -> Average -> Dense -> Output

    Where `Average` is literally just averaging the embedding vectors into one.
    """

    def __init__(self,
                 embedding: nn.Embedding,
                 num_classes: int):
        """Initliase model with embedding."""
        super().__init__()

        self.embedding = embedding

        self.output = nn.Linear(
            in_features=self.embedding.embedding_dim,
            out_features=num_classes,
        )

    # pylint: disable=arguments-differ
    def forward(self,
                utterance_tokens: Tensor,
                mask: Tensor,
                label: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Calculate model output.

        utterance_tokens: (batch, seq_len, vocab_len)
        """
        # (batch, seq_len, embedding_dim)
        embeddings = self.embedding(utterance_tokens)
        # (batch, embedding_dim)
        utterance = embeddings.mean(dim=1)
        # (batch, num_classes)
        output = self.output(utterance)

        return output, self.loss(output, mask, label)


def random_emb_simple(
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        ) -> Simple:
    """Return Simple with randomly initialised embeddings layer."""
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    return Simple(
        embedding=embedding,
        num_classes=num_classes,
    )


def glove_simple(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        ) -> Simple:
    """Return Simple with embedding layer initialised with GloVe."""
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)

    return Simple(
        embedding=embedding,
        num_classes=num_classes,
    )
