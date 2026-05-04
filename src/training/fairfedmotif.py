
from copy import deepcopy
from torch.linalg import norm
import torch
from client import Client
from server import Server
from training.gradient_helpers import flatten, unflatten



def fairfed(clients: list[Client], server: Server, communication_rounds, local_epoch, with_prototypes = False):
    """
    Excutes federated learning with incentive mechanism (gradient allocation, payoff). 
    Uses prototypes if with_prototypes is True.

    Args:
        clients (list[Client]): A list of initialized client objects participating in the training.
        server (Server): The central server coordinating the global model.
        communication_rounds (int): The total number of federated training rounds.
        local_epoch (int): The number of local training epochs each client performs per round.
    """
    
    if with_prototypes:
        num_motifs_per_client = []
        for client in clients:
            client.construct_motifs()
            num_motifs_per_client.append(len(client.motif_count.keys()))
        server.init_client_diversity(num_motifs_per_client)

    server.init_client_values(len(clients))
    
    # update client's internal reputation history
    for i in range(len(clients)):
            clients[i].reputation.append(server.client_values[i])

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        if c_round == 1: # TODO: optional to start at same point
            for client in clients:
                client.download_weights_from_server(server)

        if with_prototypes:
            for client in clients:
                    client.prototype_update()
            
            if c_round == 1: 
                server.aggregate_prototype(clients)
            else:
                server.aggregate_prototype_by_client_value(clients)

        client_gradients = []
        for client in clients:
            old_model = deepcopy(client.model)

            if with_prototypes:
                client.train_with_prototypes(server)
            else:
                client.local_train(local_epoch)

            new_model = deepcopy(client.model)

            
            old_params = dict(old_model.named_parameters())
            new_params = dict(new_model.named_parameters())

            local_gradient = []
            for param_name in server.W.keys():
                if param_name in new_params:
                    # calculate gradient strictly for the shared layers
                    grad = new_params[param_name].data - old_params[param_name].data
                    local_gradient.append(grad)

            flattened = flatten(local_gradient)
            norm_value = norm(flattened) + 1e-7
            local_gradient = unflatten(torch.div(flattened, norm_value), local_gradient)
            
            client_gradients.append(local_gradient)
        
        server.calculate_gradients(client_gradients)
        server.update_client_values(client_gradients)
       
        server.allocate_payoff(clients)

        # update client's internal reputation history
        for i in range(len(clients)):
            clients[i].reputation.append(server.client_values[i])

        reward_gradients_per_client = server.allocate_gradients()

        # allocate reward gradient to each client
        for i, client in enumerate(clients):
            for param_idx, param_name in enumerate(server.W.keys()):
                client.W[param_name].data.add_(reward_gradients_per_client[i][param_idx])
        
        # update central model
        server.update_weights()
