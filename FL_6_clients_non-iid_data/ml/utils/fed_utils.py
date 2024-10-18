from torch.utils.data import random_split
from ml.fl.client import Client
from torch.utils.data import Subset
import numpy as np


# each client has one of three outputs on its data
def create_fed_clients(trainset, nclients):
    """
    Διαχωρισμός των δεδομένων του trainset σε clients βάσει της κλάσης Target.
    """
    # Διαχωρισμός των δεδομένων βάσει της κλάσης
    targets = np.array(trainset.labels)
    dropout_indices = np.where(targets == 0)[0]  # Ετικέτα "Dropout" ως 0
    graduate_indices = np.where(targets == 2)[0]  # Ετικέτα "Graduate" ως 2
    enrolled_indices = np.where(targets == 1)[0]  # Ετικέτα "Enrolled" ως 1

    client_list = []

    # Clients 1 και 2 για Dropout
    dropout_split = len(dropout_indices) // 2
    client_list.append(Client(0, Subset(trainset, dropout_indices[:dropout_split])))
    client_list.append(Client(1, Subset(trainset, dropout_indices[dropout_split:])))

    # Clients 3, 4 και 5 για Graduate
    graduate_split = len(graduate_indices) // 3
    client_list.append(Client(2, Subset(trainset, graduate_indices[:graduate_split])))
    client_list.append(Client(3, Subset(trainset, graduate_indices[graduate_split:2 * graduate_split])))
    client_list.append(Client(4, Subset(trainset, graduate_indices[2 * graduate_split:])))

    # Client 6 για Enrolled
    client_list.append(Client(5, Subset(trainset, enrolled_indices)))

    print(f'Successfully created {len(client_list)} clients with custom data splits.')

    #for i, client in enumerate(client_list):
     #   client_targets = [trainset.labels[idx] for idx in client.dataset.indices]
      #  unique, counts = np.unique(client_targets, return_counts=True)
       # print(f'Client {i + 1}: {len(client.dataset)} samples, Label Distribution: {dict(zip(unique, counts))}')

    return client_list

#difrent length of data in clients
# def create_fed_clients(trainset, nclients):
#     """
#     Splitting provided trainset into nclients respective
#     clients objects based on specific proportions.
#     """
#     total_data = len(trainset)
#     proportions = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]
#     split_list = [int(total_data * prop) for prop in proportions]
#
#     client_data = random_split(trainset, split_list)
#     client_list = []
#     for i, dataset in enumerate(client_data):
#         new_client = Client(i, dataset)
#         client_list.append(new_client)
#         #print(f'Client {i + 1} created with {len(dataset)} samples')
#
#     print(f'Successfully created {len(client_list)} clients')
#
#     return client_list
def initialize_fed_clients(client_list, args, model):
    params = {'epochs': args.epochs, 'lr': args.lr, 'device': args.device, 'test_size': args.test_size, 'batch_size': args.batch_size, 'criterion': args.criterion, 'optimizer': args.optimizer}
    new_client_list = []
    for client in client_list:
        client.init_parameters(params, model)
        new_client_list.append(client)

    print(f'Successfully initialized {len(new_client_list)} clients.')

    return new_client_list