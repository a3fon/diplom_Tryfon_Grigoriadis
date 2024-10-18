from torch.utils.data import DataLoader
from ml.utils.train_utils import train, test
from typing import Optional, Callable, List, Tuple, OrderedDict, Union
import torch
import numpy as np
from ml.fl.aggregation.aggregator import Aggregator
from ml.fl.selectors import RandomSelector
#from ml.fl.selectors import ClusterBasedSelector
from ml.fl.selectors import AccuracySelector
from ml.fl.selectors import PowerOfChoiceSelector
from ml.fl.selectors import ImportanceSamplingSelector
class Server:
    def __init__(self, args, testset, model):
        self.dataset = testset
        self.testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        self.model = model
        self.device = args.device

        # Get Criterion
        from ml.utils.helpers import get_criterion
        self.criterion = get_criterion(args.criterion)

        #Initialize Aggregator
        self.aggregator = Aggregator(aggregation_alg=args.aggregator , params=None)
        print(f"Aggregation algorithm: {repr(self.aggregator)}")

        # Initialize Selector
        #self.selector = RandomSelector(args.fraction)

        #self.selector = ClusterBasedSelector(num_clusters=1)

        #self.selector = AccuracySelector(args.fraction)

        #self.selector = PowerOfChoiceSelector(num_clients_to_select=3, candidate_pool_size=6)

        self.selector = ImportanceSamplingSelector(args.fraction)



        print("Successfully initialized FL Server")
  
    def update(self, client_list, args):
        # Perform training for every client
        global_model_state = self.model.state_dict()
        #selected_clients = self.selector.sample_clients(client_list)

        #selected_clients = self.selector.select_clients(client_list, args.fraction)

        #selected_clients = self.selector.sample_clients(client_list)

        selected_clients = self.selector.sample_clients(client_list, global_model_state)

        for cl in selected_clients:#selected_clients
            train_history = train(cl.model,cl.train_loader, cl.device, cl.criterion, cl.optimizer, cl.epochs,False)
            # Evaluate each client on the respective testset
            acc, f1 = test(cl.model,cl.test_loader,cl.criterion, cl.device)
            #print(f'Client ID: {cl.id}  --- Test Acc: {acc}  Test F1: {f1}.')

        client_list = self.perform_federated_aggregation(client_list, selected_clients, self.aggregator)

        return client_list

        
    def evaluate(self):
        acc, f1 = test(self.model, self.testloader, self.criterion, self.device)
        return acc, f1
    
    
    def set_server_parameters(self,parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict(), strict=True)

    def get_server_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def perform_federated_aggregation(self,client_list, selected_clients, aggregator):
        
        server_model_params = self.get_server_parameters()
        results: List[Tuple[List[np.ndarray], int]] = []
        for client in selected_clients:
            results.append((client.get_parameters(), len(client.dataset)))
        
        aggregated_params = aggregator.aggregate(results, server_model_params)
        
        self.set_server_parameters(aggregated_params)

        for cl in client_list:
            cl.set_parameters(aggregated_params)

        return client_list