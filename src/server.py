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

        self.vocab = {} # motif_key: count_across_all_clients
        self.num_client = {} # motif_key: num_of_clients with motif_key
        self.global_prototype = {} # motif_key: prototype
    
    def aggregate_prototype(self, clients):
        """
        Aggregates client prototypes into a global prototype weighted by motif frequency.
        """
        # clear state from previous federated rounds
        self.clear_prototypes()

        # populate vocab + num_client
        for client in clients:
            for key, count in client.motif_count.items():
                if key not in self.vocab:
                    self.vocab[key] = count
                    self.num_client[key] = 1
                else:
                    self.vocab[key] += count
                    self.num_client[key] += 1
        
        # construct global prototype using pure frequency weighting
        for client in clients:
            for key, count in client.motif_count.items():
                
                # weight is simply (local_count / total_global_count)
                weight = count / self.vocab[key] 
                weighted_client_proto = weight * client.prototype[key]
                
                if key not in self.global_prototype:
                    self.global_prototype[key] = weighted_client_proto
                else:    
                    self.global_prototype[key] += weighted_client_proto
                    
        # extract data 
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data

    def clear_prototypes(self):
        self.vocab = {} 
        self.num_client = {} 
        self.global_prototype = {} 

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
