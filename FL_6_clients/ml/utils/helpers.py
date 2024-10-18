import torch
import copy
from logging import INFO
import numpy as np

def get_optim(model,
              optim_name: str = "adam",
              lr: float = 1e-3):
    """Returns the specified optimizer for the model defined as torch module."""
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"The specified optimizer: {optim_name} is not currently supported.")
    return optimizer


def get_criterion(crit_name: str = "cross_entropy"):
    """Returns the specified loss function."""
    if crit_name == "mse":
        criterion = torch.nn.MSELoss()
    elif crit_name == "l1":
        criterion = torch.nn.L1Loss()
    elif crit_name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif crit_name == 'nlloss':
        criterion = torch.nn.NLLLoss()
    else:
        raise NotImplementedError(f"The specified criterion: {crit_name} is not currently supported.")
    return criterion