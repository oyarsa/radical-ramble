from typing import Optional
import torch


class BaseModel(torch.nn.Module):
    _loss = torch.nn.CrossEntropyLoss()

    def set_weights(self, weights: torch.Tensor) -> None:
        self._loss = torch.nn.CrossEntropyLoss(weight=weights)

    def loss(self,
             prediction: torch.Tensor,
             true: Optional[torch.Tensor],
             ) -> Optional[torch.Tensor]:
        if true is None:
            return None
        return self._loss(prediction, true)
