"CLI script for running the module."
from typing import Optional, NamedTuple
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import incubator.datasets.meld_linear_text_dataset as mltd
import incubator.data as data
from incubator.models.simple_classifier import glove_simple_classifier
from incubator.train import train
from incubator import util


class Dataloaders(NamedTuple):
    "Training and Dev dataloaders"
    trainloader: DataLoader # type: ignore
    devloader: Optional[DataLoader] # type: ignore


class ModelData(NamedTuple):
    "Structure for holding a model and its suitable data loaders"
    model: nn.Module # type: ignore
    data: Dataloaders


def load_mltd(args: argparse.Namespace) -> Dataloaders:
    "Loads data for a MeldLinearText dataset"
    train_data = mltd.MeldLinearTextDataset(
        data=Path(args.train_data),
        mode=args.mode,
    )

    trainloader = mltd.meld_linear_text_daloader(
        dataset=train_data,
        batch_size=args.batch_size,
    )

    if args.dev_data:
        dev_data = mltd.MeldLinearTextDataset(args.dev_data, mode=args.mode)
        devloader: Optional[DataLoader] = mltd.meld_linear_text_daloader( # type: ignore
            dataset=dev_data,
            batch_size=args.batch_size,
        )
    else:
        devloader = None

    return Dataloaders(trainloader=trainloader, devloader=devloader)


def load_data(args: argparse.Namespace) -> Dataloaders:
    "Loads data with the appropriate data format for the model"
    mltd_based = ['simple_glove']

    if args.model in mltd_based:
        return load_mltd(args)

    print(f'Unknown model "{args.model}"')
    sys.exit(1)


def get_model_and_data(args: argparse.Namespace) -> ModelData:
    "Returns the initialised model and appropriate dataset"
    if args.mode == 'sentiment':
        num_classes = len(data.sentiments)
    else:
        num_classes = len(data.emotions)

    loaders = load_data(args)

    if args.model == 'simple_glove':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mltd.MeldLinearTextDataset)

        model = glove_simple_classifier(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
        )

    return ModelData(model=model, data=loaders)


def train_model(args: argparse.Namespace) -> nn.Module: # type: ignore
    """
    Instantiates and initialises the model given in `args`. Also loads
    the appropriate dataset, using the paths given. Then trains the model.

    The training loop uses `CrossEntropyLoss` as criterion. Outputs accuracy
    and loss statistics for the training dataset and (optionally) a dev
    dataset. Progress in each epoch is tracked via a progress bar.
    """
    model_data = get_model_and_data(args)

    model = train(
        model=model_data.model,
        num_epochs=args.num_epochs,
        trainloader=model_data.data.trainloader,
        devloader=model_data.data.devloader,
        gpu=args.gpu,
    )

    if args.output:
        torch.save(model, args.output)

    return model


def train_arguments(parser: argparse.ArgumentParser) -> None:
    "Adds arguments to a training command"
    defaults = {
        'glove_path': Path('./data/glove/glove.6B.50d.txt'),
        'glove_dim': 50,
    }

    parser.add_argument('--model', required=True,
                        help='Model to train')
    parser.add_argument('--mode', default='emotion',
                        help='Mode (emotion or sentiment)')
    parser.add_argument('--train_data', help='Path to training data',
                        required=True)
    parser.add_argument('--dev_data', help='Path to dev data')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of batches in training')
    parser.add_argument('--glove_path', default=defaults['glove_path'],
                        help='Path to GloVe')
    parser.add_argument('--glove_dim', type=int, default=defaults['glove_dim'],
                        help='GloVe vector dim')
    parser.add_argument('--glove_train', action='store_true',
                        help='Whether to train GloVe embeddings')
    parser.add_argument('--output', help='Output path for saved model')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which GPU to use. Defaults to CPU.')

def eval_arguments(parser: argparse.ArgumentParser) -> None:
    "Adds arguments to an evaluation command"
    defaults = {
        'glove_path': Path('./data/glove/glove.6B.50d.txt'),
        'glove_dim': 50,
    }

    parser.add_argument('--model', required=True,
                        help='Model to train')
    parser.add_argument('--mode', default='emotion',
                        help='Mode (emotion or sentiment)')
    parser.add_argument('--eval_data', help='Path to evaluation data',
                        required=True)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of batches in evaluation')
    parser.add_argument('--glove_path', default=defaults['glove_path'],
                        help='Path to GloVe')
    parser.add_argument('--glove_dim', type=int, default=defaults['glove_dim'],
                        help='GloVe vector dim')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which GPU to use. Defaults to CPU.')

def main() -> None:
    "Main function. Parses CLI arguments for train/eval commands"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Task to perform', dest='command')

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_arguments(train_parser)

    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_arguments(eval_parser)

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args)
    elif args.command == 'eval':
        print('Evaluation not implemented yet')



if __name__ == "__main__":
    util.set_seed(1000)
    main()
