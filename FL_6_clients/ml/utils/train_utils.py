import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Union
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score

def train(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    epochs: int,
    print_local = False
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    print_local : Bool
        Print info or not during training
    """
    model.train()
    train_loss = []
    train_acc = []
    train_f1 = []
    train_history = {'acc':train_acc, 'loss':train_loss, 'f1':train_f1}
    for i in range(epochs + 1):
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        acc, f1 = test(model,trainloader, criterion, device)
        train_history['acc'].append(acc)
        train_history['f1'].append(f1)
        train_history['loss'].append(loss)
        if print_local:
            print(f'Epoch {i} - Train Loss: {loss} - Train Acc: {acc} - Train F1: {f1}')
    
    return train_history




def test(
    model: nn.Module, testloader: DataLoader, criterion: torch.nn.CrossEntropyLoss, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    model : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    loss = 0.0
    model.eval()
    accs = []
    f1s = []
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            accs.append(accuracy_score(labels.cpu(), predicted.cpu()))
            f1s.append(f1_score(labels.cpu(), predicted.cpu(), average='macro'))

    acc = mean(accs)
    f1 = mean(f1s)
    return acc, f1