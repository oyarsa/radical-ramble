"ContextualSimple model and associated helper functions"
from typing import Union, TextIO, Optional, Tuple
from pathlib import Path
import torch.nn as nn
from torch import Tensor

from incubator.glove import load_glove
from incubator.data import Vocabulary
from incubator.models.base_model import BaseModel


class ContextualSimple(BaseModel):
    """
    Simplest possible model from sequence of tokens to a class output

        TokenIds -> Embedding -> Average -> Dense -> Output

    Where `Average` is literally just averaging the embedding vectors into one.
    """
    def __init__(self,
                 embedding: nn.Embedding,
                 num_classes: int):
        super().__init__()
        self.set_reduction('none')

        self.embedding = embedding

        self.output = nn.Linear(
            in_features=self.embedding.embedding_dim,
            out_features=num_classes,
        )

    # pylint: disable=arguments-differ
    def forward(self,
                dialogue_tokens: Tensor,
                masks: Tensor,
                labels: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        dialogue_tokens: (batch, nutterances, seq_len)
        labels: Optional(batch, nutterances)
        """
        # (batch, nutterances, seq_len, embedding_dim)
        embeddings = self.embedding(dialogue_tokens)
        # (batch, nutterances, embedding_dim)
        utterances = embeddings.mean(dim=2)
        # (batch, nutterances, nclasses)
        outputs = self.output(utterances)

        return outputs, self.loss(outputs, masks, labels)


def glove_contextual_simple(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
        freeze: bool = True,
        saved_glove_file: Optional[Path] = None,
        ) -> ContextualSimple:
    "ContextualSimple with embedding layer initialised with GloVe"
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
        saved_glove_file=saved_glove_file,
    )

    embedding = nn.Embedding.from_pretrained(glove, freeze=freeze)

    return ContextualSimple(
        embedding=embedding,
        num_classes=num_classes,
    )
