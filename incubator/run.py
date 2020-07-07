"CLI script for running the module."
from typing import Optional, NamedTuple
import argparse
import sys
from pathlib import Path

import configargparse
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import wandb  # type: ignore

import incubator.datasets.meld_linear_text_dataset as mltd
import incubator.datasets.meld_contextual_text_dataset as mctd
import incubator.data as data
from incubator.models.simple import glove_simple
from incubator.models.linear_rnn import glove_linear_lstm
from incubator.models.linear_cnn import glove_linear_cnn
from incubator.models.linear_cnn_rnn import glove_linear_cnn_lstm
from incubator.models.contextual_simple import glove_contextual_simple
from incubator.train import train
from incubator import util
from incubator.config import defaults
from incubator.models.base_model import BaseModel


class Dataloaders(NamedTuple):
    "Training and Dev dataloaders"
    trainloader: DataLoader
    devloader: Optional[DataLoader]


class ModelData(NamedTuple):
    "Structure for holding a model and its suitable data loaders"
    model: BaseModel
    data: Dataloaders


def load_mctd(args: argparse.Namespace) -> Dataloaders:
    "Loads data for a MeldLinearText dataset"
    train_data = mctd.MeldContextualTextDataset(
        data=args.train_data,
        mode=args.mode,
    )

    trainloader = mctd.meld_contextual_text_daloader(
        dataset=train_data,
        batch_size=args.batch_size,
    )

    if args.dev_data:
        dev_data = mctd.MeldContextualTextDataset(
            data=args.dev_data,
            mode=args.mode,
            vocab=train_data.vocab,
        )
        devloader = mctd.meld_contextual_text_daloader(
            dataset=dev_data,
            batch_size=args.batch_size,
        )
    else:
        devloader = None

    return Dataloaders(trainloader=trainloader, devloader=devloader)


def load_mltd(args: argparse.Namespace) -> Dataloaders:
    "Loads data for a MeldLinearText dataset"
    train_data = mltd.MeldLinearTextDataset(
        data=args.train_data,
        mode=args.mode,
    )

    trainloader = mltd.meld_linear_text_daloader(
        dataset=train_data,
        batch_size=args.batch_size,
    )

    if args.dev_data:
        dev_data = mltd.MeldLinearTextDataset(
            data=args.dev_data,
            mode=args.mode,
            vocab=train_data.vocab,
        )
        devloader = mltd.meld_linear_text_daloader(
            dataset=dev_data,
            batch_size=args.batch_size,
        )
    else:
        devloader = None

    return Dataloaders(trainloader=trainloader, devloader=devloader)


def load_data(args: argparse.Namespace) -> Dataloaders:
    "Loads data with the appropriate data format for the model"
    mltd_based = ['simple', 'linear_rnn', 'linear_cnn', 'linear_cnn_rnn']
    mctd_based = ['contextual_simple']

    if args.model in mltd_based:
        return load_mltd(args)
    if args.model in mctd_based:
        return load_mctd(args)

    print(f'Unknown model "{args.model}"')
    sys.exit(1)


def get_model_and_data(args: argparse.Namespace) -> ModelData:
    "Returns the initialised model and appropriate dataset"
    if args.mode == 'sentiment':
        num_classes = len(data.sentiments)
    else:
        num_classes = len(data.emotions)

    loaders = load_data(args)

    if not args.saved_glove_file:
        glove_name = f'glove.{args.glove_dim}d.{args.model}.p'
        saved_glove_file = Path(args.train_data).parent / glove_name
    else:
        saved_glove_file = args.saved_glove_file

    filters = [int(f) for f in args.cnn_filters.split(',')]

    model: BaseModel

    if args.model == 'simple':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mltd.MeldLinearTextDataset)

        model = glove_simple(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
            saved_glove_file=saved_glove_file,
        )
    elif args.model == 'linear_rnn':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mltd.MeldLinearTextDataset)

        model = glove_linear_lstm(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
            saved_glove_file=saved_glove_file,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_num_layers=args.rnn_num_layers,
            bidirectional=args.rnn_bidirectional,
            rnn_dropout=args.rnn_dropout,
        )
    elif args.model == 'linear_cnn':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mltd.MeldLinearTextDataset)

        model = glove_linear_cnn(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
            saved_glove_file=saved_glove_file,
            filters=filters,
            out_channels=args.cnn_out_channels,
            dropout=args.dropout,
        )
    elif args.model == 'linear_cnn_rnn':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mltd.MeldLinearTextDataset)

        model = glove_linear_cnn_lstm(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
            saved_glove_file=saved_glove_file,
            filters=filters,
            out_channels=args.cnn_out_channels,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_num_layers=args.rnn_num_layers,
            bidirectional=args.rnn_bidirectional,
            rnn_dropout=args.rnn_dropout,
        )
    elif args.model == 'contextual_simple':
        train_data = loaders.trainloader.dataset
        assert isinstance(train_data, mctd.MeldContextualTextDataset)

        model = glove_contextual_simple(
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
            num_classes=num_classes,
            vocab=train_data.vocab,
            freeze=not args.glove_train,
            saved_glove_file=saved_glove_file,
        )
    else:
        print(f'Unknown model "{args.model}"')
        sys.exit(1)

    print(f'\n{model}\n')

    return ModelData(model=model, data=loaders)


def train_model(args: argparse.Namespace) -> nn.Module:
    """
    Instantiates and initialises the model given in `args`. Also loads
    the appropriate dataset, using the paths given. Then trains the model.

    The training loop uses `CrossEntropyLoss` as criterion. Outputs accuracy
    and loss statistics for the training dataset and (optionally) a dev
    dataset. Progress in each epoch is tracked via a progress bar.
    """
    model_data = get_model_and_data(args)

    model = model_data.model
    wandb.watch(model)

    weights: Optional[torch.Tensor]
    if args.mode == 'emotion':
        weights = torch.tensor([4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0])
    else:
        weights = None

    model = train(
        model=model,
        num_epochs=args.num_epochs,
        trainloader=model_data.data.trainloader,
        devloader=model_data.data.devloader,
        gpu=args.gpu,
        log_interval=args.log_interval,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        weights=weights,
    )

    torch.save(model.state_dict(), Path(wandb.run.dir) / 'model.pt')
    if args.output:
        torch.save(model, args.output)

    return model


def train_arguments(parser: argparse.ArgumentParser) -> None:
    "Adds arguments to a training command"
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='Configuration file')
    parser.add_argument('--model', default='simple',
                        help='Model to train')
    parser.add_argument('--mode', default='emotion',
                        help='Mode (emotion or sentiment)')
    parser.add_argument('--train_data', default=defaults['train_data'],
                        help='Path to training data')
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Optimiser (Adam) learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay factor for L2 regularisation')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which GPU to use. Defaults to CPU')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Number of batches between each report to WandB')
    parser.add_argument('--saved_glove_file',
                        help='Path to pre-processed GloVe file')
    parser.add_argument('--cnn_filters', default='3',
                        help='Filter sizes for CNN, separated by commas')
    parser.add_argument('--cnn_out_channels', type=int, default=100,
                        help='Number of output channels for CNN')
    parser.add_argument('--rnn_hidden_size', type=int, default=200,
                        help='Dimensionality of the RNN hidden layer / output')
    parser.add_argument('--rnn_bidirectional', action='store_true',
                        help='Whether to have the RNN be bidirectional')
    parser.add_argument('--rnn_num_layers', type=int, default=1,
                        help='Number of stacked RNN layers')
    parser.add_argument('--rnn_dropout', type=float, default=0,
                        help='RNN dropout')
    parser.add_argument('--dropout', type=float, default=0,
                        help='General dropout')


def eval_arguments(parser: argparse.ArgumentParser) -> None:
    "Adds arguments to an evaluation command"
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='Configuration file')
    parser.add_argument('--model', default='simple',
                        help='Model to train')
    parser.add_argument('--mode', default='emotion',
                        help='Mode (emotion or sentiment)')
    parser.add_argument('--eval_data', default=defaults['eval_data'],
                        help='Path to evaluation data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of batches in evaluation')
    parser.add_argument('--glove_path', default=defaults['glove_path'],
                        help='Path to GloVe')
    parser.add_argument('--glove_dim', type=int, default=defaults['glove_dim'],
                        help='GloVe vector dim')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which GPU to use. Defaults to CPU.')
    parser.add_argument('--saved_glove_file',
                        help='Path to pre-processed GloVe file')


def main() -> None:
    "Main function. Parses CLI arguments for train/eval commands"
    parser = configargparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Task to perform', dest='command')

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_arguments(train_parser)

    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_arguments(eval_parser)

    args = parser.parse_args()

    wandb.init(project='incubator')
    wandb.config.update(args)

    if args.command == 'train':
        train_model(args)
    elif args.command == 'eval':
        print('Evaluation not implemented yet')


if __name__ == "__main__":
    util.set_seed(1000)
    main()
