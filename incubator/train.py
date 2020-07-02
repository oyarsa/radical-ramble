"Training loop for torch models"
from typing import Optional, NamedTuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import wandb
from sklearn.metrics import precision_recall_fscore_support


class EpochResults(NamedTuple):
    "Results for a given epoch"
    accuracy: float
    loss: float


def train_epoch(
        epoch: int,
        model: nn.Module, # type: ignore
        trainloader: DataLoader, # type: ignore
        criterion: nn.CrossEntropyLoss,
        optimiser: optim.Optimizer, # type: ignore
        device: torch.device,
        log_interval: int = 10,
        ) -> EpochResults:
    "Performs training pass for an epoch. Reports loss and accuracy"
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
        inputs = batch.utterance_tokens.to(device)
        labels = batch.labels.to(device)

        # forward + backward + optim pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        running_acc += (outputs.argmax(1) == labels).sum().item()
        running_len += outputs.shape[0]

        cur_loss = running_loss / running_len
        cur_acc = running_acc / running_len

        pbar.set_description(f'#{epoch} [Train] acc: {cur_acc:.3f}'
                             f' loss: {cur_loss:.3f}')
        pbar.update(1)

        if i % log_interval == 0:
            wandb.log({'Train accuracy': cur_acc, 'Train loss': cur_loss})


    return EpochResults(
        accuracy=running_acc / len(trainloader),
        loss=running_loss / len(trainloader),
    )


def dev_epoch(epoch: int,
              model: nn.Module, # type: ignore
              devloader: DataLoader, # type: ignore
              criterion: nn.CrossEntropyLoss,
              device: torch.device,
              log_interval: int = 10,
              ) -> EpochResults:
    "Performs evaluation of dev dataset for an epoch"
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
            inputs = batch.utterance_tokens.to(device)
            labels = batch.labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(1)

            running_loss += loss.item()
            running_acc += (predictions == labels).sum().item()
            running_len += outputs.shape[0]

            y_pred = torch.cat((y_pred, predictions.cpu()))
            y_true = torch.cat((y_true, labels.cpu()))

            cur_loss = running_loss / running_len
            cur_acc = running_acc / running_len
            f1 = precision_recall_fscore_support(
                y_true,
                y_pred,
                zero_division=1,
                average='weighted',
            )[2]

            pbar.set_description(f'#{epoch} [Dev  ] acc: {cur_acc:.3f} '
                                 f'loss: {cur_loss:.3f} '
                                 f'f1: {f1:.3f}')
            pbar.update(1)

            if i % log_interval == 0:
                wandb.log({'Dev accuracy': cur_acc, 'Dev loss': cur_loss})

    return EpochResults(
        accuracy=running_acc / len(devloader),
        loss=running_loss / len(devloader),
    )


def train(model: nn.Module, # type: ignore
          trainloader: DataLoader, # type: ignore
          devloader: Optional[DataLoader] = None, # type: ignore
          num_epochs: int = 10,
          gpu: int = -1,
          learning_rate: float = 0.01,
          verbose=False,
          log_interval: int = 10,
          ) -> nn.Module: # type: ignore
    "Performs training loop"
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_results = train_epoch(epoch, model, trainloader, criterion,
                                    optimiser, device, log_interval)

        if devloader:
            dev_results = dev_epoch(epoch, model, devloader, criterion, device,
                                    log_interval)

        if verbose:
            print(train_results)
            print(dev_results)

    return model
