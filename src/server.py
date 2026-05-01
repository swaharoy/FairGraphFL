import random
import torch


class Server():
    """
    Central server module for Federated Learning.

    The server holds the global master model. In baseline training (selftrain), 
    it simply acts as a common initialization point. In full federated learning, 
    it handles the aggregation of client weights.

    Attributes:
        model (torch.nn.Module): The global graph neural network model.
        W (dict): A dictionary referencing the model's named parameters.
    """
    def __init__(self, model, device):
        """
        Initializes the Server.

        Args:
            model (torch.nn.Module): The global GNN model instance.
            device (torch.device): The device (CPU or CUDA) to host the model on.
        """
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}  # points to named_parameters()
    
    def rand_sample_clients(self, all_clients, frac):
        """
        Randomly samples a subset of clients for a federated communication round.

        Args:
            all_clients (list[Client]): The complete pool of available clients.
            frac (float): The fraction (0.0 to 1.0) of clients to sample.

        Returns:
            list[Client]: A randomly selected subset of clients.
        """
        return random.sample(all_clients, int(len(all_clients) * frac))
    
    def aggregate_weights(self, clients):
        """
        Performs FedAvg aggregation.

        Calculates the weighted average of the local models from the selected clients 
        based on their local dataset sizes, and updates the global server model.

        Args:
            clients (list[Client]): The list of clients that participated in the current round.
        """
        total_size = 0
        for client in clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in clients]), dim=0), total_size).clone()
