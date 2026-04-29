import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.linalg import norm
from copy import deepcopy
from utils import *
import matplotlib.pyplot as plt
import scipy.stats


### BASELINE RUNS (can recover for other baselines in past commits)
def run_selftrain_GC(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return allAccs

def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


### PROTOTYPES (no incentive mechanism)

def run_prototype(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    """
        Does not use reputation when aggregating prototypes. Does not do gradient / payoff allocation either.
    """
    #l = []
    selected_clients = clients
    for client in selected_clients:
        client.motif_construction()
        
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        #l1 = 0
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        if c_round == 1:
            for client in selected_clients:
                
                client.prototype_update()
        server.aggregate_prototype(selected_clients)
        

        for client in selected_clients:
            client.download_code(server)
            client.prototype_train(server)
            loss, acc = client.evaluate()
            #l1 += loss
        #l.append(l1 / len(clients))


        for client in selected_clients:
            client.clear_prototype()
        server.clear_prototype()
    #l = np.array(l)
    #np.save('FG.npy', l)
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print('w/o reput')
    return frame

def run_prototype_with_reputation_weighted_aggregation(clients, server, COMMUNICATION_ROUNDS, device, dp=False, samp=None, frac=1.0):
    """
        Doesn't do model allocation / payoff, but uses reputation when aggregating prototypes.
        Each global prototype is the weighted sum of client prototypes where each weight is
        the agent value for that specfic prototype. Thus, more reputable cleints for each specific prototype
        will influence the update.
    """
    for client in clients:
        client.motif_construction()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.prototype_update()

        if c_round == 1:
            server.aggregate_prototype(clients)
        else:
            server.aggregate_prototype_weighted_by_client_reput_per_motif(clients)
       
        phis = torch.zeros(len(clients))
        for i, client in enumerate(clients):
            phis[i] = client.cosine_similar(server) 

        for client in clients:
            client.download_code(server)
            if not c_round == 1:
                for _ in range(1):
                    client.prototype_train(server)
            
        server.update_reput(clients)
        
        for client in clients:
            client.clear_prototype()
        server.clear_prototype()
        
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print('reput2')
    return frame


### INCENTIVE MECHANISM

def run_incentive_mech_wo_prototypes(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    """
        Runs incentive mechanism without prototypes.
        - rewards global gradients based on rs
        - simple payoff aggregation (not accountin for delayed contribution)
        - since not using prototypes, does not use diversity
    """
    rs = torch.zeros(len(clients))
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)

    for c_round in range(1, communication_rounds+1):
        for i in range(len(clients)):
            clients[i].reput = rs[i]
        
        if c_round % 50 ==0:
            print(f"  > round {c_round}")
        
        # calculate the local gradient
        gradients = []
        for client in clients:
            old = deepcopy(client.model)

            client.local_train(local_epoch)
            new = deepcopy(client.model)
           
            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
            gradient = []
            for i in range(2, 14): # 2-13 are conv layers 
                gradient.append(local_gradient[i])

            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7

            gradient = unflatten(torch.div(flattened, norm_value), gradient)
            gradients.append(gradient)


        # calculate the global gradient
        global_gradient = [torch.zeros(param.shape).to(server.device) for param in server.model.parameters()]
        for gradient, weight in zip(gradients, rs):
            if weight < 0:
                continue
            else:
                add_gradient_updates(global_gradient, gradient, weight)
        
        # normalize global gradient (eq 13)
        s = torch.sum(F.relu(rs)).item()
        for g in global_gradient:
            g = torch.div(g, s)

        # calculate the reputation
        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)
        for i, client in enumerate(clients):
            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]

        rs = torch.div(rs, rs.sum())

        # simple payoff aggregation
        for i, client in enumerate(clients):
            client.payoff += rs[i]
        
        # distribute the global gradient
        q_ratios = torch.tanh(0.5 * rs)
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        for i in range(len(clients)):
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='all')
            
            for j, k in enumerate(server.W.keys()):
                clients[i].W[k] = clients[i].W[k] + reward_gradient[j]

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)

    print("Payoffs: \n")
    for client in clients:
        print(client.payoff_history)        

def run_incentive_mech_with_prototypes(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    """
        Runs Algorithm 1 of paper.

        Incentive Mechanism:
        - rewards global gradients based on rs
        - calculated payoff accounting for delayed contribution

        Prototypes (for model quality enhancement)
        - uses diversity in valuation function
        - global prototypes aggregated by client reputation
    """
    # Initialize agent values
    rs = torch.zeros(len(clients))
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)

    # Calculate client diversity
    diversity = [] 
    for client in clients:
        client.motif_construction()
        diversity.append(len(client.prototype.keys()))
    diversity = torch.tensor(diversity, dtype = torch.long)
    # modified from paper, dividing by max num of prototypes rather than total unique
    # intuitively, the client with the most motifs will have their value diminished the least
    diversity = diversity / torch.max(diversity) 


    for c_round in range(1, communication_rounds+1):
        for i in range(len(clients)):
            clients[i].reput = rs[i]
            clients[i].reputation.append(rs[i])

        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

       

        # update global prototypes
        if c_round == 1: 
            # initializes global prototypes 

            for client in clients: 
                    client.prototype_update()

            # TODO: what is the "paper" method for initializing global prototypes
            # clients with a high freq of a motif has more weight in the prototype avg
            server.aggregate_prototype(clients)                  
        else: 
            server.aggregate_prototype_by_client_reput(clients)

        # update local model
        client_gradients = []
        for client in clients:
            old_model = deepcopy(client.model)

            client.download_code(server)

            # trains local model AND updates local prototypes
            client.prototype_train(server)

            new_model = deepcopy(client.model)

            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]
            
            gradient = []
            for i in range(2, 14): # 2-13 are conv layers 
                gradient.append(local_gradient[i])
            
            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7
            gradient = unflatten(torch.div(flattened, norm_value), gradient)
            
            client_gradients.append(gradient)

        # calculate the global gradient (eq 13)
        global_gradient = [torch.zeros(param.shape).to(server.device) for param in server.model.parameters()]
        for gradient, weight in zip(client_gradients, rs):
            if weight < 0:
                continue
            else:
                add_gradient_updates(global_gradient, gradient, weight)

        # normalize global gradient (eq 13)
        rs_sum = torch.sum(F.relu(rs)).item()
        for gradient in global_gradient:
            gradient.div_(rs_sum)

        # update reputation
        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in client_gradients], device=server.device)
        for i, client in enumerate(clients):
            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]
            rs[i] *= diversity[i]
        rs = torch.div(rs, rs.sum())

        # money payoff 
        allocate_payoff(clients, rs)

        # add current agent values to client reputation
        for i, client in enumerate(clients):
            client.reputation.append(rs[i])

        # allocate gradients to clients (\beta = 0.5)
        q_ratios = torch.tanh(0.5 * rs)
        q_ratios /= torch.max(q_ratios)
        print(f"q_rations {q_ratios}")

        for i in range(len(clients)):
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='all')

            for j, k in enumerate(server.W.keys()):
                clients[i].W[k] = clients[i].W[k] + reward_gradient[j]
        
        print('finish c_round of inc mech w proto')

        for client in clients:
            client.clear_prototype()
        server.clear_prototype()


    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    for client in clients:
        print(client.payoff)
    fs = frame.style.apply(highlight_max).data

    print(fs)
    print('incentive_mech_with_prototypes')

    return frame


### HELPERS

def compare_local_motif_freq_distribution(client1, client2):
    """
    Computes structural similarity and topological discrepancy between two clients 
    by analyzing their local motif frequency distributions.

    Args:
        client1 (Client_GC)
        client2 (Client_GC)
    
    Returns:
        tuple: 
            - wasserstein_dist (float): Statistical distance (EMD) between 
              structural distributions.
            - avg_diff (float): Average L1 discrepancy (Topological Volume Mismatch) 
              per motif type.
    """

    # Use the pre-aggregated counts (Stage 1's source)
    m1 = client1.motif_count
    m2 = client2.motif_count
    
    # Get all unique motifs across both clients
    all_motifs = set(m1.keys()).union(set(m2.keys()))
    
    freq1 = []
    freq2 = []
    total_diff = 0
    
    for motif in all_motifs:
        f1 = m1.get(motif, 0)
        f2 = m2.get(motif, 0)
        
        freq1.append(f1)
        freq2.append(f2)
        total_diff += abs(f1 - f2)

    # Metrics
    D = scipy.stats.wasserstein_distance(freq1, freq2)
    avg_diff = total_diff / len(all_motifs)
    
    # Plotting (only need to do this once!)
    plt.plot(freq1, label=client1.name)
    plt.plot(freq2, label=client2.name)
    plt.savefig('comparison.png')
    
    return D, avg_diff

    
def allocate_payoff(clients, rs):
    """
        Calculates payoff per client. 
        If agent value (rs[i]) > 0, calculates past contribution using reputation history.
        Curr communication round stored in client.payoff
        Also appended to client.payoff_history
    """

    total_payoff_c_round = 1e-9

    for i, client in enumerate(clients):
        prev_rounds_avg = torch.tensor(client.reputation).mean() 

        if rs[i] < 0:
            client.payoff = rs[i]
        else:
            past_contribtuion = torch.max(torch.tensor([rs[i] - prev_rounds_avg, 0]))
            client.payoff = rs[i] + past_contribtuion

        total_payoff_c_round +=  client.payoff
    
    for client in clients:
        client.payoff /= total_payoff_c_round
        client.payoff_history.append(client.payoff)
    
    
        

    

    
