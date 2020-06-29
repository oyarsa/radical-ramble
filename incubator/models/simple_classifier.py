from typing import Union, TextIO
from pathlib import Path
import torch.nn as nn
from torch import Tensor

from incubator.glove import load_glove
from incubator.data import Vocabulary

class SimpleClassifier(nn.Module): # type: ignore
    def __init__(self,
                 embedding: nn.Embedding,
                 num_classes: int):
        super(SimpleClassifier, self).__init__()

        self.embedding = embedding

        self.output = nn.Linear(
            in_features=self.embedding.embedding_dim,
            out_features=num_classes,
        )

    # pylint: disable=arguments-differ
    def forward(self, utteranceTokens: Tensor) -> Tensor:
        """
            utteranceTokens: (batch, seq_len, vocab_len)
        """
        # (batch, seq_len, embedding_dim)
        embeddings = self.embedding(utteranceTokens)
        # (batch, embedding_dim)
        utterance = embeddings.mean(dim=1)
        # (batch, num_classes)
        output = self.output(utterance)

        return output

def random_emb_simple_classifier(
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
    ) -> SimpleClassifier:
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    return SimpleClassifier(
        embedding=embedding,
        num_classes=num_classes,
    )

def glove_simple_classifier(
        glove_path: Union[Path, TextIO],
        glove_dim: int,
        num_classes: int,
        vocab: Vocabulary,
    ) -> SimpleClassifier:
    glove = load_glove(
        input_file=glove_path,
        glove_dim=glove_dim,
        vocab=vocab,
    )

    embedding = nn.Embedding.from_pretrained(glove)

    return SimpleClassifier(
        embedding=embedding,
        num_classes=num_classes,
    )
