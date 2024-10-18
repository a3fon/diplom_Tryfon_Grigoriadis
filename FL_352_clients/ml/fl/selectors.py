import random
import torch.nn as nn
import numpy as np


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
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s)")
        return sampled_clients


class L2Selector:
    def __init__(self, fraction):
        self.fraction = fraction
        self.dist_function = nn.MSELoss()  # Mean Squared Error used as a proxy for L2 distance

    def sample_clients(self, client_list, global_model_state):
        """ Select clients based on the L2 norm of their model's state compared to the global model. """
        num_clients_to_select = int(self.fraction * len(client_list))
        if num_clients_to_select == 0:
            num_clients_to_select = 1

        client_distances = []

        # Calculate the L2 distance between each client's model and the global model
        for cl in client_list:
            total_distance = 0.0
            for param_tensor in cl.model.state_dict():
                client_param = cl.model.state_dict()[param_tensor]
                global_param = global_model_state[param_tensor]

                # Compute the distance for the parameter tensor
                distance = self.dist_function(client_param, global_param)
                total_distance += distance.item()

            client_distances.append((cl, total_distance))

        # Sort clients by distance and select the top ones based on the selection strategy
        client_distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)

        # Select the closest clients
        selected_clients = [client_distances[i][0] for i in range(num_clients_to_select)]

        print(f"Sampled {num_clients_to_select} client(s) based on L2 distance: {[cl.id for cl in selected_clients]}")
        return selected_clients


class CASelector:
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
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s): {sampled_clients}")
        for i in sampled_clients:
            print(i)
        return sampled_clients

class ImportanceSelectors:
    def __init__(self, fraction, importance_function=None):
        self.fraction = fraction
        self.importance_function = importance_function if importance_function is not None else self.default_importance

    def default_importance(self, client):
        # Example: Use client's last evaluated accuracy as importance
        return client.last_accuracy if client.last_accuracy is not None else 0.0

    def sample_clients(self, client_list):
        if not client_list:
            print("No clients available to sample.")
            return []

        # Calculate importance scores for each client
        importance_scores = [(client, self.importance_function(client)) for client in client_list]

        # Debug: Print importance scores before sorting
        print("Importance Scores (before sorting):", [(cl.id, score) for cl, score in importance_scores])

        # Sort clients by importance score in descending order
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        # Debug: Print sorted client IDs by importance
        print("Sorted Clients by Importance:", [(cl.id, score) for cl, score in importance_scores])

        # Select top fraction of clients based on importance
        num_selection = max(1, int(self.fraction * len(client_list)))
        sampled_clients = [client for client, _ in importance_scores[:num_selection]]

        print(f"Sampled {num_selection} client(s) based on importance: {[cl.id for cl in sampled_clients]}")
        return sampled_clients

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

class AccuracySelector:
    def __init__(self, fraction):
        self.fraction = fraction

    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []

        # Calculate validation performance for each client
        importance_weights = [self.evaluate_client_performance(cl) for cl in available_clients]#performances

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
        print(f"Importance sampling: Sampled {num_selection} client(s): {[cl.id for cl in selected_clients]}")

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