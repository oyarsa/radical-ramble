from typing import Optional
import torch
import torch.nn.functional as F


class BaseModel(torch.nn.Module):
    _loss_weights: Optional[torch.Tensor]
    _loss_reduction: str

    def __init__(self):
        super().__init__()
        self._loss_weights = None
        self._loss_reduction = 'mean'

    def set_weights(self, weights: torch.Tensor) -> None:
        self._loss_weights = weights

    def set_reduction(self, reduction: str) -> None:
        if reduction not in ['none', 'sum', 'mean']:
            raise ValueError('Invalid loss reduction')
        self._loss_reduction = reduction

    def loss(self,
             outputs: torch.Tensor,
             masks: torch.Tensor,
             labels: Optional[torch.Tensor],
             ) -> Optional[torch.Tensor]:
        if labels is None:
            return None

        if outputs.dim() == 2:  # if linear, outputs: (batch, nclasses)
            return F.cross_entropy(
                input=outputs,
                target=labels,
                weight=self._loss_weights,
                reduction=self._loss_reduction,
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
            reduction=self._loss_reduction,
        )
        losses *= masks

        return losses.sum() / masks.nonzero().shape[0]
