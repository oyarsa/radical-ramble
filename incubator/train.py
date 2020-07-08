"""Training loop for torch models."""
from typing import Optional, Tuple, cast

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import precision_recall_fscore_support
import wandb  # type: ignore

from incubator.models.base_model import BaseModel


def calc_accuracy(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        ) -> float:
    """
    Calculate accuracy given model outputs, labels and masks.

    outputs: (batch, *, nclasses)
    labels: (batch, *)
    mask: (batch, *)

    Observation: the '*' dimensions all must match.
    """
    predictions = outputs.argmax(-1)
    results = predictions == labels
    results = results * mask

    acc = results.sum().item()
    return cast(float, acc)


def get_predictions(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        outputs: torch.Tensor,
        new_true: torch.Tensor,
        mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate predictions and true labels.

    Does the right thing when it comes to masking and contextual models. That
    is, it flattens the 2-D structures from the contextual models, whilst
    ignoring the masked positions.
    """
    new_pred = outputs.argmax(-1) * mask

    if mask.dim() == 1:
        y_pred = torch.cat((y_pred, new_pred.cpu()))
        y_true = torch.cat((y_true, new_true.cpu()))
        return y_pred, y_true

    result_pred = []
    result_true = []

    for i in range(new_pred.shape[0]):
        for j in range(new_pred.shape[1]):
            if mask[i, j] == 1:
                result_pred.append(new_pred[i, j].item())
                result_true.append(new_true[i, j].item())

    result_pred = torch.tensor(result_pred)
    result_true = torch.tensor(result_true)
    return result_pred, result_true


def train_epoch(
        epoch: int,
        model: BaseModel,
        trainloader: DataLoader,
        optimiser: optim.Optimizer,
        device: torch.device,
        log_interval: int = 10,
        ) -> None:
    """Perform training pass for an epoch. Reports loss and accuracy."""
    running_loss = 0.0
    running_acc = 0.0
    running_len = 0

    pbar = tqdm.trange(len(trainloader),
                       desc=f'#{epoch} [Train] acc: 0.000 loss: 0.000',
                       leave=True)

    model.train()
    for i, batch in enumerate(trainloader):
        # zero the parameter gradients
        optimiser.zero_grad()

        # get input and gold label
        inputs = batch.tokens.to(device)
        labels = batch.labels.to(device)
        masks = batch.masks.to(device)

        # forward + backward + optim pass
        outputs, loss = model(inputs, masks, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        running_acc += calc_accuracy(outputs, labels, masks)
        running_len += masks.nonzero().shape[0]

        cur_loss = running_loss / running_len
        cur_acc = running_acc / running_len

        pbar.set_description(f'#{epoch} [Train] acc: {cur_acc:.3f}'
                             f' loss: {cur_loss:.3f}')
        pbar.update(1)

        if i % log_interval == 0:
            wandb.log({
                'Train accuracy': cur_acc,
                'Train loss': cur_loss,
            })


def dev_epoch(epoch: int,
              model: BaseModel,
              devloader: DataLoader,
              device: torch.device,
              log_interval: int = 10,
              ) -> None:
    """Perform evaluation of dev dataset for an epoch."""
    pbar = tqdm.trange(len(devloader),
                       desc=f'#{epoch} [Dev  ] acc: 0.000 loss: 0.000',
                       leave=True)

    model.eval()
    with torch.no_grad():
        running_acc = 0.0
        running_loss = 0.0
        running_len = 0

        y_pred = torch.tensor([]).long()
        y_true = torch.tensor([]).long()

        for i, batch in enumerate(devloader):
            inputs = batch.tokens.to(device)
            labels = batch.labels.to(device)
            masks = batch.masks.to(device)

            outputs, loss = model(inputs, masks, labels)

            running_loss += loss.item()
            running_acc += calc_accuracy(outputs, labels, masks)
            running_len += masks.nonzero().shape[0]

            y_pred, y_true = get_predictions(y_pred, y_true,
                                             outputs, labels, masks)

            cur_loss = running_loss / running_len
            cur_acc = running_acc / running_len
            f1_score = precision_recall_fscore_support(
                y_true,
                y_pred,
                zero_division=1,
                average='weighted',
            )[2]

            pbar.set_description(f'#{epoch} [Dev  ] acc: {cur_acc:.3f} '
                                 f'loss: {cur_loss:.3f} '
                                 f'f1: {f1_score:.3f}')
            pbar.update(1)

            if i % log_interval == 0:
                wandb.log({
                    'Dev accuracy': cur_acc,
                    'Dev loss': cur_loss,
                    'Dev f1': f1_score,
                })


def train(model: BaseModel,
          trainloader: DataLoader,
          devloader: Optional[DataLoader] = None,
          num_epochs: int = 10,
          gpu: int = -1,
          learning_rate: float = 0.01,
          weight_decay: float = 1e-5,
          verbose=False,
          log_interval: int = 10,
          weights: Optional[torch.Tensor] = None,
          ) -> nn.Module:
    """Perform training loop."""
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')

    model = model.to(device)

    if weights is not None:
        model.set_weights(weights.to(device))

    optimiser = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    for epoch in range(num_epochs):
        train_epoch(epoch, model, trainloader, optimiser, device, log_interval)

        if devloader:
            dev_epoch(epoch, model, devloader, device, log_interval)

    return model
