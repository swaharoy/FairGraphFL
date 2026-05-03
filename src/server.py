from copy import copy
import math
import random
import torch
import torch.nn.functional as F
from training.gradient_helpers import flatten, unflatten # TODO: torch built-in isntead?




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
        self.device = device
        self.model = model.to(self.device)
        self.W = {key: value for key, value in self.model.named_parameters()}  # points to named_parameters()

        self.vocab = {} # motif_key: count_across_all_clients
        self.num_client = {} # motif_key: num_of_clients with motif_key
        self.global_prototype = {} # motif_key: prototype

        self.client_values = []
        self.client_diversity = []

        self.gradients = []
    
    def init_client_values(self, num_of_clients):
        self.client_values = torch.zeros(num_of_clients)

        for i in range(num_of_clients):
            self.client_values[i] = 1 / num_of_clients

    def init_client_diversity(self, num_motifs_per_client):
        """
         modified from paper, dividing by max num of prototypes rather than total unique
         intuitively, the client with the most motifs will have their value diminished the least
        """
        self.client_diversity = torch.zeros(len(num_motifs_per_client),  dtype = torch.long)

        for i, num_motifs in enumerate(num_motifs_per_client):
            self.client_diversity[i] = num_motifs
        
        self.client_diversity =  self.client_diversity / torch.max(self.client_diversity)
        
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

    def aggregate_prototype_by_client_value(self, clients):
        """
        Aggregates client prototypes based on client value
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
        
        # construct global prototype weighting by client reputation
        weight = {}
        for i, client in enumerate(clients):
            relu_val =  F.relu(self.client_values[i]).item()

            for key, count in client.motif_count.items():
                weight[key] = weight.get(key, 0) + relu_val
                
                if key not in self.global_prototype:
                    self.global_prototype[key] = relu_val * client.prototype[key]
                else:    
                    self.global_prototype[key] += relu_val * client.prototype[key]

        for key in self.global_prototype.keys():
            self.global_prototype[key] /= max(weight[key], 1e-8)          

        # extract data 
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data

    def clear_prototypes(self):
        self.vocab = {} 
        self.num_client = {} 
        self.global_prototype = {} 

    def calculate_gradients(self, client_gradients): # TODO: actually update the weight
        """"
         Construct global server gladient by aggregating oer each client's gradient (weighted by value)
        """
        self.gradients = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        for i, client_gradient in enumerate(client_gradients):
            relu_val = max(0, self.client_values[i])
            self._aggregate_gradient_updates(client_gradient, weight= relu_val)
        
        # normalize
        client_values_sum = max(torch.sum(F.relu(self.client_values)).item(), 1e-8)
        for gradient in self.gradients:
            gradient.div_(client_values_sum)
        
    def update_client_values(self, client_gradients):
        """
            Update client value as a function of client and server gradient similarity and client diversity.
        """
        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(self.gradients), 0, 1e-10) for gradient in client_gradients], device=self.device)
        for i in range(len(client_gradients)):
            self.client_values[i] = 0.95 * self.client_values[i] + 0.05 * phis[i]
            self.client_values[i] *= self.client_diversity[i]
        self.client_values = torch.div(self.client_values, self.client_values.sum())

    def allocate_payoff(self, clients):
        """
        Calculates payoff per client. 
        If agent value (rs[i]) > 0, calculates past contribution using reputation history.
        Curr communication round appended to in client.payoff
        """

        total_payoff_c_round = 1e-9

        for i, client in enumerate(clients):
            prev_rounds_avg = torch.tensor(client.reputation).mean() 

            if self.client_values[i] < 0:
                client.payoff.append(self.client_values[i])
            else:
                past_contribtuion = torch.max(torch.tensor([ self.client_values[i] - prev_rounds_avg, 0]))
                client.payoff.append(self.client_values[i] + past_contribtuion)

            total_payoff_c_round +=  client.payoff[-1]
        
        for client in clients:
            client.payoff[-1] /= total_payoff_c_round

    def allocate_gradients(self, clients):
        """
            Mask global gradient for each client wrt client's value
        """
        q_ratios = torch.tanh(0.5 * self.client_values)
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        reward_gradient_per_client = []
        for i, client in enumerate(clients):
            reward_gradient = _mask_grad_update_by_order(self.gradients, mask_percentile=q_ratios[i], mode='all')
            reward_gradient_per_client.append(reward_gradient)
        
        return reward_gradient_per_client




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

    def _aggregate_gradient_updates(self, client_gradient, weight):
        assert len(client_gradient) == len(self.gradients), "Lengths of the two grad_updates not equal"
	
        for i, client_param in enumerate(client_gradient):
            self.gradients[i].data += client_param.data * weight

def _mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):
    if mode == 'layer':
        grad_update = copy.deepcopy(grad_update)
        mask_percentile = max(0, mask_percentile)
        for i, layer in enumerate(grad_update):   
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)
            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
                
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update
    elif mode == 'all':
        all_update_mod = torch.cat([update.data.view(-1).abs() for update in grad_update])
        mask_percentile = max(0, mask_percentile)
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)
        if mask_order == 0:
            return _mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, min(mask_order, len(all_update_mod)))
            return _mask_grad_update_by_magnitude(grad_update, topk[-1])
        

def _mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update
