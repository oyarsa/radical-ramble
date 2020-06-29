from typing import Optional, Tuple, NamedTuple
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

class EpochResults(NamedTuple):
    accuracy: float
    loss: float


def train_epoch(
        epoch: int,
        model: nn.Module, # type: ignore
        trainloader: DataLoader, # type: ignore
        criterion: nn.CrossEntropyLoss,
        optimiser: optim.Optimizer, # type: ignore
        ) -> EpochResults:
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm.trange(len(trainloader),
                       desc=f'#{epoch} [Train] acc: 0.000 loss: 0.000',
                       leave=True)

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

        cur_loss = running_loss / (i + 1)
        cur_acc = running_acc / (i + 1)

        pbar.set_description(f'#{epoch} [Train] acc: {cur_acc:.3f}'
                             f' loss: {cur_loss:.3f}')
        pbar.update(1)

    return EpochResults(
        accuracy = running_acc / len(trainloader),
        loss = running_loss / len(trainloader),
    )


def dev_epoch(epoch: int,
              model: nn.Module, # type: ignore
              devloader: DataLoader, # type: ignore
              criterion: nn.CrossEntropyLoss) -> EpochResults:
    pbar = tqdm.trange(len(devloader),
                       desc=f'#{epoch} [Dev  ] acc: 0.000 loss: 0.000',
                       leave=True)

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

            cur_loss = dev_loss / (i + 1)
            cur_acc = dev_acc / (i + 1)

            pbar.set_description(f'#{epoch} [Dev  ] acc: {cur_acc:.3f} '
                                 f'loss: {cur_loss:.3f}')
            pbar.update(1)

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

    for epoch in range(num_epochs):
        train_results = train_epoch(epoch, model, trainloader,
                                    criterion, optimiser)

        if devloader:
            dev_results = dev_epoch(epoch, model, devloader, criterion)

        terminal_width, _ = shutil.get_terminal_size()
        print('-' * terminal_width)
