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

### DOES NOT USE INCENTIVES MECHANISM

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
    # TODO: remove redundant rs from this func
    rs = torch.zeros(len(clients)) # Init value of agent
    for i in range(len(clients)):
        rs[i] = 1 / len(clients)

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
            
        rs = 0.95 * rs + 0.05 * phis
        rs = torch.clamp(rs, min=1e-3)
        rs = torch.div(rs, rs.sum())

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


### USES INCENTIVES MECHANISM

def run_protoreput3(clients, server, COMMUNICATION_ROUNDS, device, samp=None, frac=1.0):
    
    for client in clients:
        client.motif_construction()
    selected_clients = clients
    weight = {}
    for c_round in range(1, COMMUNICATION_ROUNDS+1):
        weight = {}
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in selected_clients:
                
                client.prototype_update()
            server.aggregate_prototype(selected_clients)
            for client in selected_clients:
                for motif in client.prototype.keys():
                    client.rs[motif] = 0.05 * F.cosine_similarity(client.prototype[motif], server.global_prototype[motif], 0, 1e-10)
                    client.rs[motif] = torch.clamp(client.rs[motif], min=1e-3)

                    
        else:
            server.reput3_prototype(selected_clients)
            for client in selected_clients:
                for motif in client.prototype.keys():
                    client.rs[motif] = 0.95 * client.rs[motif] + 0.05 * F.cosine_similarity(client.prototype[motif], server.global_prototype[motif], 0, 1e-10)
                    client.rs[motif] = torch.clamp(client.rs[motif], min=1e-3)
        #reput update
        for client in clients:
            for motif in client.prototype.keys():
                if motif not in weight.keys():
                    weight[motif] = client.rs[motif]
                else:
                    weight[motif] += client.rs[motif]

                pass
        for client in clients:
            for motif in client.prototype.keys():
                client.rs[motif] /= weight[motif]
                client.rs[motif] = client.rs[motif]

        for client in clients:
            reput = torch.zeros(len(client.rs.keys()))
            for i, motif in enumerate(client.rs.keys()):
                reput[i] = client.rs[motif]
            client.reput = torch.mean(reput)
            #client.reput = np.mean(list(client.rs.values()))
        weight = 0
        for client in clients:
            weight += client.reput
        for client in clients:
            client.reput /= weight

        for client in selected_clients:
            client.download_code(server)
            client.prototype_train(server)
        # aggregation
        #aggregated_gradient = [torch.zeros(param.shape).to(server.args.device) for param in server.model.parameters()]
        # old = deepcopy(server.model)
        # for k in server.W.keys():
        #     server.W[k].data = torch.sum(torch.stack([torch.mul(client.W[k].data, client.reput) for client in selected_clients]), dim=0).clone()
        # new = deepcopy(server.model)
        # gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
        # print('finish1')
        
        # # distribute model parameters
        # rs = torch.zeros(len(selected_clients))
        # for i in range(len(selected_clients)):
        #     rs[i] = selected_clients[i].reput
        # q_ratios = torch.tanh(1.5 * rs)
        # q_ratios /= torch.max(q_ratios)
        # for i, client in enumerate(selected_clients):
        #     reward_gradient = mask_grad_update_by_order(gradient, mask_percentile=q_ratios[i], mode='layer')
        #     client.gconvNames = server.W.keys()
        #     for j, key in enumerate(server.W.keys()):
        #         client.W[key] = client.W[key] + reward_gradient[j]
        # print('finish2')
        for client in selected_clients:
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
    print('reput3')
    return frame

def run_reput(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    selected_clients = clients
    rs = torch.zeros(len(clients))
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)
    # for client in selected_clients:
    #     print(client.train_size)
    for c_round in range(1, communication_rounds+1):
        for i in range(len(selected_clients)):
            selected_clients[i].reput = rs[i]
        
        if c_round % 50 ==0:
            print(f"  > round {c_round}")
        

        # calculate the local gradient
        gradients = []
        for client in selected_clients:
            old = deepcopy(client.model)
            if client.reput > 0:
                client.local_train(local_epoch)
                new = deepcopy(client.model)
                #client.model.load_state_dict(old.state_dict())
            else:
                client.local_train(local_epoch)
                new = deepcopy(client.model)
            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
            gradient = []
            for i in range(2, 14):
            # for i in range(len(local_gradient)):
                gradient.append(local_gradient[i])

            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7

            gradient = unflatten(torch.div(flattened, norm_value), gradient)


            gradients.append(gradient)


        # calculate the global gradient
        # old = deepcopy(server.model)
        #print(old.parameters())

        # for k in server.W.keys():
        #     server.W[k].data = torch.sum(torch.stack([torch.mul(client.W[k].data, F.relu(client.reput)) for client in selected_clients]), dim=0) / torch.sum(F.relu(torch.tensor(rs))).clone()
        # new = deepcopy(server.model)
        # global_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
        global_gradient = [torch.zeros(param.shape).to(server.device) for param in server.model.parameters()]
        for gradient, weight in zip(gradients, rs):
            if weight < 0:
                continue
            else:
                add_gradient_updates(global_gradient, gradient, weight)
        
        s = torch.sum(F.relu(rs)).item()
        for g in global_gradient:
            g = torch.div(g, s)

        # calculate the reputation
        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)
        #print(phis)
        for i, client in enumerate(clients):
            
            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]
        # 1. set all the reputations to be positive
        #rs = torch.clamp(rs, min=1e-3)

        rs = torch.div(rs, rs.sum())

        # DW
        for i, client in enumerate(clients):
            rs[i] = len(client.dataLoader['train'])
            # ablation
        rs /= rs.sum()



        
        for i, client in enumerate(selected_clients):
            client.payoff += rs[i]
        
        # distribute the global gradient
        q_ratios = torch.tanh(0.5 * rs)
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        for i in range(len(selected_clients)):
            # print(global_gradient)
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='all')
            

            for j, k in enumerate(server.W.keys()):
                
                selected_clients[i].W[k] = selected_clients[i].W[k] + reward_gradient[j]

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    for client in selected_clients:
        print(client.payoff)        

def run_reput2(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    rs = torch.zeros(len(clients))
    proto = []
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)
    for client in clients:
        client.motif_construction()
        print(len(client.prototype.keys()))
        proto.append(len(client.prototype.keys()))
    

    proto = torch.tensor(proto, dtype = torch.long)
    proto = proto / torch.max(proto)

    selected_clients = clients
    weight = {}
    for c_round in range(1, communication_rounds+1):

        for i in range(len(selected_clients)):
            selected_clients[i].reput = rs[i]
        weight = {}
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        

        # global aggregate update    
        if c_round == 1:
            for client in selected_clients:
                
                client.prototype_update()
            server.aggregate_prototype(selected_clients)
            # for client in selected_clients:
            #     for motif in client.prototype.keys():
            #         client.rs[motif] = 0.05 * F.cosine_similarity(client.prototype[motif], server.global_prototype[motif], 0, 1e-10)
            #         client.rs[motif] = torch.clamp(client.rs[motif], min=1e-3)

            #print(len(server.global_prototype.keys()))       
        else:
            server.reput3_prototype(selected_clients)
            
            
            print(len(server.global_prototype.keys()))
        # print(len(server.global_prototype.keys())) 
        # update the local model and calculate the local gradient
        gradients = []
        for client in selected_clients:
            old = deepcopy(client.model)
            client.download_code(server)
            client.prototype_train(server)
            new = deepcopy(client.model)
            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
            gradient = []
            for i in range(2, 14):
            # for i in range(len(local_gradient)):
                gradient.append(local_gradient[i])
            
            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7

            gradient = unflatten(torch.div(flattened, norm_value), gradient)
            gradients.append(gradient)
        global_gradient = [torch.zeros(param.shape).to(server.device) for param in server.model.parameters()]
        for gradient, weight in zip(gradients, rs):
            if weight < 0:
                continue
            else:
                add_gradient_updates(global_gradient, gradient, weight)




        s = torch.sum(F.relu(rs)).item()
        for gradient in global_gradient:
            gradient = torch.div(gradient, s)



        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)
        for i, client in enumerate(clients):
            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]
            # ablation
            rs[i] *= proto[i]
        
        #rs = torch.div(rs, rs.sum())
        for i, client in enumerate(clients):
            rs[i] = len(client.dataLoader['train'])
            # ablation
        rs /= rs.sum()
        

        # money payoff
        for i, client in enumerate(selected_clients):
            if c_round == 1:
                client.payoff += rs[i]
            else:
                k = np.mean(np.array(client.reputation))
                client.payoff += rs[i] + torch.max(torch.tensor([rs[i] - k, 0]))
            client.reputation.append(rs[i])

        # distribute model parameters
        
        q_ratios = torch.tanh(0.5 * rs)
        q_ratios /= torch.max(q_ratios)

        for i in range(len(selected_clients)):
            
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='layer')
            # ablation
            reward_gradient = global_gradient
            for j, k in enumerate(server.W.keys()):
                selected_clients[i].W[k] = selected_clients[i].W[k] + reward_gradient[j]
        
        print('finish2')

        for client in selected_clients:
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
    print('reput3')

    return frame

def run_reput3(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    rs = torch.zeros(len(clients))
    proto = []
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)
    for client in clients:
        client.motif_construction()
        print(len(client.prototype.keys()))
        proto.append(len(client.prototype.keys()))

    proto = torch.tensor(proto, dtype = torch.long)
    proto = proto / torch.max(proto)

    selected_clients = clients
    weight = {}
    for c_round in range(1, communication_rounds+1):

        for i in range(len(selected_clients)):
            selected_clients[i].reput = rs[i]
        weight = {}
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        

        # global aggregate update    
        if c_round == 1:
            for client in selected_clients:

                client.download_from_server(server)
                client.prototype_update()
            server.aggregate_prototype(selected_clients)
            # for client in selected_clients:
            #     for motif in client.prototype.keys():
            #         client.rs[motif] = 0.05 * F.cosine_similarity(client.prototype[motif], server.global_prototype[motif], 0, 1e-10)
            #         client.rs[motif] = torch.clamp(client.rs[motif], min=1e-3)

                    
        else:
            server.reput3_prototype(selected_clients)
        # print(len(server.global_prototype.keys())) 
        # update the local model and calculate the local gradient
        old = deepcopy(server.model)
        gradients = []
        for client in selected_clients:
            old = deepcopy(client.model)
            client.download_code(server)
            client.prototype_train(server)
            new = deepcopy(client.model)
            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
            gradients.append(local_gradient)
        for i, client in enumerate(selected_clients):
            if i == 1:
                server.model = client.model
            else:
                for (param_1, param_2) in zip(server.model.parameters(), client.model.parameters()):
                    param_1 = param_1 + param_2 * F.relu(rs[i])
        new = deepcopy(server.model)
        global_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
         

        s = torch.sum(F.relu(rs)).item()
        for gradient in global_gradient:
            gradient = torch.div(gradient, s)



        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)
        for i, client in enumerate(clients):
            phis = torch.tensor([F.cosine_similarity(flatten(gradients[i]), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)

            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]
            # ablation
            rs[i] = proto[i]
        
        rs = torch.div(rs, rs.sum())

        # money payoff
        for i, client in enumerate(selected_clients):
            if c_round == 1:
                client.payoff += rs[i]
            else:
                k = np.mean(np.array(client.reputation))
                client.payoff += rs[i] - k
            client.reputation.append(rs[i])

        # distribute model parameters
        
        q_ratios = torch.tanh(2 * rs)
        q_ratios /= torch.max(q_ratios)

        for i in range(len(selected_clients)):
            
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='layer')
            # ablation
            # reward_gradient = global_gradient
            for j, k in enumerate(server.W.keys()):
                selected_clients[i].W[k] = selected_clients[i].W[k] + reward_gradient[j]
        

        print('finish2')


        for client in selected_clients:
            client.clear_prototype()
        server.clear_prototype()


    for client in clients:
        client.download_from_server(server)
    correct = 0
    total = 0
    for client in clients:
        _, _, acc_sum, n_graphs = client.evaluate()
        print(acc_sum)
        print(n_graphs)
        correct += acc_sum
        total += n_graphs
        # print(n_graphs)
        # print(acc_sum)
    acc = correct / total
    print(f'total acc={acc}')

    
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
    print('reput3')

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

    

    
    
        

    

    
