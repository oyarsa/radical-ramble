from typing import Optional, Tuple, NamedTuple
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


class EpochResults(NamedTuple):
    accuracy: float
    loss: float


def train_epoch(
        model: nn.Module, # type: ignore
        trainloader: DataLoader, # type: ignore
        criterion: nn.CrossEntropyLoss,
        optimiser: optim.Optimizer, # type: ignore
        ) -> EpochResults:
    running_loss = 0.0
    running_acc = 0.0

    for i, batch in enumerate(trainloader):
        # zero the parameter gradients
        optimiser.zero_grad()

        # get input and gold label
        inputs = batch.utterance_tokens
        labels = batch.labels

        # forward + backward + optim pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        running_acc += (outputs.argmax(1) == labels).sum().item()

    return EpochResults(
        accuracy = running_acc / len(trainloader),
        loss = running_loss / len(trainloader),
    )


def dev_epoch(model: nn.Module, # type: ignore
              devloader: DataLoader, # type: ignore
              criterion: nn.CrossEntropyLoss) -> EpochResults:
    with torch.no_grad():
        dev_acc = 0.0
        dev_loss = 0.0
        for i, batch in enumerate(devloader):
            inputs = batch.utterance_tokens
            labels = batch.labels

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            dev_loss += loss.item()
            dev_acc += (outputs.argmax(1) == labels).sum().item()

        return EpochResults(
            accuracy = dev_acc / len(devloader),
            loss = dev_loss / len(devloader),
        )


def train(model: nn.Module, # type: ignore
          trainloader: DataLoader, # type: ignore
          devloader: Optional[DataLoader] = None, # type: ignore
          num_epochs: int = 10,
          ) -> None:
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    print()
    for epoch in range(num_epochs):
        train_results = train_epoch(model, trainloader, criterion, optimiser)

        print(f'[{epoch}/{num_epochs}] train_acc: {train_results.accuracy} '
              f'train_loss {train_results.loss}', end=' ')

        if devloader:
            dev_results = dev_epoch(model, devloader, criterion)
            print(f'dev_acc: {dev_results.accuracy} '
                  f'dev_loss {dev_results.loss}', end='')

        print()
