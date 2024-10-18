import random
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from ml.utils.train_utils import train, test

class RandomSelector:
    def __init__(self, fraction):
        self.fraction = fraction


    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []
        num_selection = int(self.fraction * len(available_clients))
        if num_selection == 0:
            num_selection = 1
        if num_selection > len(available_clients):
            num_selection = len(available_clients)
        sampled_clients = random.sample(available_clients, num_selection)
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s): {[cl.id for cl in sampled_clients]}")
        return sampled_clients

class AccuracySelector:
    def __init__(self, fraction):
        self.fraction = fraction

    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []

        # Calculate validation performance for each client
        importance_weights = [self.evaluate_client_performance(cl) for cl in available_clients]

        # Convert performances to importance weights (lower performance -> higher importance)
        # Assuming higher performance is better, invert the performances
        #max_performance = max(performances)
        #importance_weights = [max_performance - p for p in performances]

        # Normalize importance weights to sum to 1
        total_importance = sum(importance_weights)
        if total_importance == 0:
           probabilities = [1 / len(importance_weights)] * len(importance_weights)
        else:
           probabilities = [w / total_importance for w in importance_weights]

        # Number of clients to sample
        num_selection = int(self.fraction * len(available_clients))
        num_selection = max(1, num_selection)

        # Sample clients based on importance probabilities
        selected_clients = np.random.choice(available_clients, num_selection, replace=False, p=probabilities)

        # Print selected clients with their probabilities
        #print(f"Importance sampling: Sampled {num_selection} client(s): {[cl.id for cl in selected_clients]}")
        # print("Selected client details:")
        # for i, cl in enumerate(selected_clients):
        #     print(f"Client ID: {cl.id}, Importance Weight: {probabilities[available_clients.index(cl)]:.4f}")

        return selected_clients

    def evaluate_client_performance(self, client):
        # Evaluate the client's model on their validation data
        acc, _ = client.evaluate(client.test_loader)
        return acc

class PowerOfChoiceSelector:
    def __init__(self, num_clients_to_select: int, candidate_pool_size: int):
        self.num_clients_to_select = num_clients_to_select
        self.candidate_pool_size = candidate_pool_size

    def sample_clients(self, client_list):
        """Select clients using the Power-of-Choice strategy."""
        selected_clients = []

        available_clients = client_list[:]  # Make a copy of the client list

        while len(selected_clients) < self.num_clients_to_select and available_clients:
            # Randomly sample a pool of candidate clients
            pool_size = min(self.candidate_pool_size, len(available_clients))
            candidate_pool = random.sample(available_clients, pool_size)

            # Evaluate candidates based on desired criteria
            candidate_performance = [(client, client.evaluate(client.test_loader)) for client in candidate_pool]

            # Check if any candidates are available
            if not candidate_performance:
                break

            # Sort candidates by performance
            candidate_performance.sort(key=lambda x: x[1], reverse=True)  # Assuming higher accuracy is better

            # Select the best candidate
            best_client = candidate_performance[0][0]
            selected_clients.append(best_client)
            available_clients.remove(best_client)

        return selected_clients

class ImportanceSamplingSelector:
    def __init__(self, fraction, device='cpu'):
        self.fraction = fraction
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device

    def calculate_gradient_norm(self, client, global_model_state):
        client.model.load_state_dict(global_model_state)
        client.model.to(self.device)
        client.model.eval()

        # Get client's batch of data points
        data, target = client.get_data_points()
        data, target = data.to(self.device), target.to(self.device)

        # Zero gradients
        client.model.zero_grad()

        # Forward pass
        output = client.model(data)

        # Calculate loss
        loss = self.loss_function(output, target)

        # Backward pass to calculate gradients
        loss.backward()

        # Calculate gradient norm
        grad_norm = sum(p.grad.norm().item() for p in client.model.parameters() if p.grad is not None)

        return grad_norm

    def sample_clients(self, client_list, global_model_state):
        if not client_list:
            print("No clients available to sample.")
            return []

        # Calculate importance weights based on gradient norm
        importance_weights = [
            (client, self.calculate_gradient_norm(client, global_model_state))
            for client in client_list
        ]

        # Sort clients based on gradient norm (importance)
        importance_weights.sort(key=lambda x: x[1], reverse=True)

        # Calculate number of clients to select
        num_selection = max(1, int(self.fraction * len(client_list)))

        # Select top clients based on importance
        sampled_clients = [client for client, _ in importance_weights[:num_selection]]

        print(f"Sampled {num_selection} client(s) based on importance sampling: {[cl.id for cl in sampled_clients]}")
        return sampled_clients