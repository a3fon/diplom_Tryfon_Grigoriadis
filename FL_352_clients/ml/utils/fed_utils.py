from torch.utils.data import random_split, DataLoader
from ml.fl.client import Client
from torch.utils.data import DataLoader, Subset

def create_fed_clients(trainset, nclients):
    """
    Splitting provided trainset into nclients respective
    clients objects.
    """
    data_per_client = len(trainset)/nclients
    split_list = [data_per_client]*nclients
    split_list = [int(i) for i in split_list]
    client_data = random_split(trainset, split_list)
    client_list = []
    for i, dataset in enumerate(client_data):
        new_client = Client(i, dataset)
        client_list.append(new_client)

    print(f'Successfully created {len(client_list)} clients.')

    return client_list
def initialize_fed_clients(client_list, args, model):
    params = {'epochs': args.epochs, 'lr': args.lr, 'device': args.device, 'test_size': args.test_size, 'batch_size': args.batch_size, 'criterion': args.criterion, 'optimizer': args.optimizer}
    new_client_list = []
    for client in client_list:
        client.init_parameters(params, model)
        new_client_list.append(client)

    print(f'Successfully initialized {len(new_client_list)} clients.')

    return new_client_list