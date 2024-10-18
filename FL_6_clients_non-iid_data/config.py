import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y", "")


def str2none(v):
    if v in ('None', 'none'):
        return None
    return v


def centralized_args():
    parser = argparse.ArgumentParser(description="Perform Centralized Learning.")
    parser.add_argument('--seed', type=int, default=1, help="The seed to initialize the random generators.")
    parser.add_argument('--batch_size', type=int, default=64, help="The batch size to load datasets")
    parser.add_argument('--model_name', type=str, default='cnn',choices=['cnn'], help="The model to use for training.")
    parser.add_argument('--epochs', type=int, default=10, help="The number of epochs for model training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="The learning rate to use.")
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help="The optimizer to use.")
    parser.add_argument('--criterion', type=str, default="cross_entropy",choices=["mse", "cross_entropy", "nlloss", "l1"], help="The criterion to use.")
    parser.add_argument('--device', type=str, default='cpu', help="Choose between cpu and cuda")
    parser.add_argument('--early_stopping', type=str2bool, default=True, help="Whether to use early stopping.")
    parser.add_argument('--patience', type=int, default=50, help="The patience value for early stopping.")

    args = parser.parse_args()
    return args


def federated_args():
    parser = argparse.ArgumentParser(description="Perform Federated Learning.")
    parser.add_argument('--seed', type=int, default=1, help="The seed to initialize the random generators.")
    parser.add_argument('--batch_size', type=int, default=64, help="The batch size to load datasets")
    parser.add_argument('--model_name', type=str, default='cnn',choices=['cnn'], help="The model to use for training.")
    parser.add_argument('--epochs', type=int, default=5, help="The number of epochs for model training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="The learning rate to use.")
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help="The optimizer to use.")
    parser.add_argument('--criterion', type=str, default="cross_entropy",choices=["mse", "cross_entropy", "nlloss", "l1"], help="The criterion to use.")
    parser.add_argument('--device', type=str, default='cpu', help="Choose between cpu and cuda")
    parser.add_argument('--early_stopping', type=str2bool, default=True, help="Whether to use early stopping.")
    parser.add_argument('--patience', type=int, default=50, help="The patience value for early stopping.")
    parser.add_argument('--test_size', type=float, default=0.25, help="The fraction of samples to use for testing.")
    parser.add_argument('--clients', type=int, default=6, help="The number of clients taking part in the federated learning process.")
    parser.add_argument('--fl_rounds', type=int, default=5, help="The number of federated rounds for model training.")
    parser.add_argument('--fraction', type=float, default=1, help="The fraction of clients to consider for local model training.")
    parser.add_argument('--aggregator', type=str, default="fedprox", help="The federated aggregation algorithm.")

    args = parser.parse_args()
    return args
