import torch

from typing import Dict, Tuple, List, Union, Optional, Any
from collections import OrderedDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ml.utils.train_utils import train, test

import numpy as np

class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.trainloader = None
        self.testloader = None
        self.model = None
        self.optimizer = None
        self.epochs = None
        self.lr = None
        self.criterion = None
        self.device = None
        self.test_size = None
        self.batch_size = None


    def init_parameters(self, params: Dict[str, Union[bool, str, int, float]], model):  # default parameters
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.model = model
        self.device = params["device"]
        self.test_size = params['test_size']
        self.batch_size = params['batch_size']

        # Get Criterion
        from ml.utils.helpers import get_criterion
        self.criterion = get_criterion(params['criterion'])

        # Get Optimizer
        from ml.utils.helpers import get_optim
        self.optimizer = get_optim(model, params['optimizer'], self.lr)

        # Train - Test Split
        train_size = int(len(self.dataset) * (1 - self.test_size))
        val_size = len(self.dataset) - train_size  # Ensure the total sum equals the dataset size
        # print(train_size, val_size)
        train_set, val_set = random_split(self.dataset, [train_size, val_size])
        # train_set, val_set = random_split(self.dataset, [int(len(self.dataset)*(1 - self.test_size)), int(len(self.dataset)*self.test_size)])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)



    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict(), strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def update(self):
        train_history = train(self.model,self.train_loader, self.device, self.criterion, self.optimizer, self.epochs,False)

    def evaluate(self, test_loader):
        acc, f1 = test(self.model,test_loader,self.criterion, self.device)
        return acc, f1

    def get_data_points(self):
        """Get a batch of data points for gradient calculation."""
        return next(iter(self.train_loader))
