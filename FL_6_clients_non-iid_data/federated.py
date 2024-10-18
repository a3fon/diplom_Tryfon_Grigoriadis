import random
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from config import federated_args, str2bool
from ml.utils.train_utils import train, test
import time
start_time = time.time()

# Load arguments
args = federated_args()
print(f"Script arguments: {args}\n")

# Enable Cuda if available
if torch.cuda.is_available():
    device = args.device
else:
    device = 'cpu'

# ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
from dataset.load_dataset import load_data
#trainset, testset = load_data()
trainset, testset, label_encoder = load_data()

# Print Dataset Details
print("== Predict Students' Dropout and Academic Success ==")
in_dim = 1
#num_classes = len(torch.unique(torch.as_tensor(trainset.Target)))
num_classes = len(torch.unique(trainset.labels))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset)}')
print(f'Test Samples: {len(testset)}')
print(f'Num of Clients: {args.clients}')
print("===============")

fractions = [6/6,5/6,4/6,3/6,2/6,1/6]#4/6,3/6,2/6,1/6
for fraction in fractions:
    print(f"Running Federated Learning with fraction = {fraction}")
    args.fraction = fraction  # Update the fraction in the arguments

    # Create Clients - each client has its own id, trainloader, testloader, model, optimizer
    from ml.utils.fed_utils import create_fed_clients
    client_list = create_fed_clients(trainset, args.clients)


    # Initialize model, optimizer, criterion
    # Get Model
    from ml.models.nn import ReceptorNet
    model = ReceptorNet()
    model.to(device)

    # Initialize Fed Clients
    from ml.utils.fed_utils import initialize_fed_clients
    client_list = initialize_fed_clients(client_list, args, copy.deepcopy(model))

    # Initiazlize Server with its own strategy, global test, global model, global optimizer, client selection
    from ml.fl.server import Server
    Fl_Server = Server(args, testset, copy.deepcopy(model))

    total_accuracy = 0.0
    total_f1 = 0.0

    for round in range(args.fl_rounds):
        print(f"FL Round: {round}")
        client_list = Fl_Server.update(client_list,args)
        acc, f1 = Fl_Server.evaluate()
        total_accuracy += acc
        total_f1 += f1
        print(f'Round {round} - Server Accuracy: {acc}, Server F1: {f1}.')

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    average_accuracy = total_accuracy / (args.fl_rounds )
    average_f1 = total_f1 / (args.fl_rounds )
    print(f'Average Accuracy over all rounds: {average_accuracy}')
    print(f'Average F1 Score over all rounds: {average_f1}')