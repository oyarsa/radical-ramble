import torch.nn as nn
from torch import Tensor

class SimpleClassifier(nn.Module): # type: ignore
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super(SimpleClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.output = nn.Linear(
            in_features=embedding_dim,
            out_features=num_classes,
        )

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
