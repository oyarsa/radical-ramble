"""Base model is the base class for all models."""
from typing import Optional
import torch
import torch.nn.functional as F  # NOQA


class BaseModel(torch.nn.Module):
    """
    BaseModel is a base class for all models.

    BaseModel contains configuration (weights) for the loss function.
    This allows reuse of the training code for different types of
    models (linear vs contextual), as we decide how to perform the loss
    calculation depending on the dimensionality of the inputs.
    """

    _loss_weights: Optional[torch.Tensor]

    def __init__(self):
        """Initialise weights to `None`."""
        super().__init__()
        self._loss_weights = None

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set loss function weights."""
        self._loss_weights = weights

    def loss(self,
             outputs: torch.Tensor,
             masks: torch.Tensor,
             labels: Optional[torch.Tensor],
             ) -> Optional[torch.Tensor]:
        """
        Calculate loss function, depending on model type.

        If the model is linear, output will have shape (batch, nclasses) and
        labels will have shape (batch,). In this case, we perform CrossEntropy
        loss directly, using the configured weights.

        If the mode is contextual, output will be (batch, nutterance, nclasses)
        and labels will be (batch, nutterances). In this case we have to
        rehsape both to (batch * nutterance, *) so that we can perform the
        CrossEntropy. However, we will ask for it to return all the losses
        instead of automatically reducing them with their mean, like in the
        simple case. This will allow us to mask the losses and calculate the
        mean ourselves.
        """
        if labels is None:
            return None

        if outputs.dim() == 2:  # if linear, outputs: (batch, nclasses)
            return F.cross_entropy(
                input=outputs,
                target=labels,
                weight=self._loss_weights,
                reduction='mean',
            )

        # if contextual, outputs: (batch, nutterance, nclasses)

        # (batch * nutterances, nclasses)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        # (batch * nutterances)
        labels = labels.reshape(-1)
        # (batch * nutterances)
        masks = masks.reshape(-1)

        losses = F.cross_entropy(
            input=outputs,
            target=labels,
            weight=self._loss_weights,
            reduction='none',
        )
        losses *= masks

        return losses.sum() / masks.nonzero().shape[0]
