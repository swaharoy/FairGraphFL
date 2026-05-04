import os
import argparse
import random
import torch
import numpy as np
from pathlib import Path

from models import GIN, serverGIN
from net import GCN
from client import Client
from server import Server

from dataset.setup_dataset import setup_dataset
from training.selftrain import selftrain
from training.fedavg import fedavg
from training.fairfedmotif import fairfed
from metrics import collect_all_metrics

def parse_args():
    """
    Parses command line arguments for the federated learning simulation.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
   
    parser.add_argument('--dataset', help='specify the dataset',
                        type=str, default='Cora')
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--method', help='FL training framework',
                    type=str, default='selftrain')
    parser.add_argument('--partition', help='subgraph partitioning method',
                        type=str, default='random')
    parser.add_argument('--skip_client', help='skip one of the clients', type = bool, default= False )
    parser.add_argument('--skip_client_idx', help='skip client at idx, must be in [0, num_clients)', type =int, default= 1 )
    parser.add_argument('--model', help='model type', type =str, default= "GIN" )

    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.85)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    return args

def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The chosen seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_experiment_prefix(args):
    """Generates a unique experiment prefix based on parameters."""

    if args.method == "central":
        name = f"{args.dataset}_{args.method}_s{args.seed}"
    else:
        name =  f"{args.dataset}_n{args.num_clients}_{args.partition}_{args.method}_s{args.seed}"

        
    return name

def save_experiment_dataframes(args, server_stats, client_stats, client_incentives):
    """
    Saves the 3 metrics dataframes to Google Drive with an informative name.
    """

    prefix = get_experiment_prefix(args)
    save_dir = os.path.join(args.outbase, prefix)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the dataframes
    server_stats.to_csv(os.path.join(save_dir, 'server.csv'), index=False)
    client_stats.to_csv(os.path.join(save_dir, 'client.csv'), index=False)
    if client_incentives:
        client_incentives.to_csv(os.path.join(save_dir, 'incentives.csv'), index=False)
    
    # Save a hyperparameters text file for easy scanning later
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
            
    print(f"Successfully saved results to: {save_dir}")


def init_clients(subgraphs, num_classes, num_node_features, args) -> list[Client]:
    """
    Initializes the local models and configurations for all federated clients.

    Args:
        subgraphs (list): List of Data objects representing local partitions.
        num_classes (int): Number of total target classes.
        num_node_features (int): Number of input features per node.
        args (argparse.Namespace): Simulation arguments.

    Returns:
        list[Client]: A list of instantiated Client objects.
    """

    clients = []
    
    sorted_subgraphs = sorted(subgraphs, key=lambda sg: sg.num_inter_edges)

    for idx, subgraph in enumerate(sorted_subgraphs):
        
        if args.model == "GCN":
            print("client model: GCN")
            model = GCN(nfeat= num_node_features, nhid= args.hidden, nclass= num_classes, nlayer= args.nlayer,dropout= args.dropout)
        else:
            print("client model: GIN")
            model = GIN(nfeat= num_node_features, nhid= args.hidden, nclass= num_classes, nlayer= args.nlayer, dropout= args.dropout)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client(client_id=idx, model= model, subgraph= subgraph, optimizer=optimizer, args=args))

    return clients

def init_server(global_graph, num_classes, num_node_features, args):
    """
    Initializes the central server that handles global aggregation.

    Args:
        args (argparse.Namespace): Simulation arguments.

    Returns:
        Server: The instantiated central server object.
    """
    if args.model == "serverGIN": 
        print("server model: server GIN")
        model = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    elif args.model == "GCN":
        print("server model: GCN")
        model =  GCN(nfeat= num_node_features, nhid= args.hidden, nclass= num_classes, nlayer= args.nlayer,dropout= args.dropout)
    else:
        print("server model: GIN")
        model = GIN(nfeat= num_node_features, nhid= args.hidden, nclass= num_classes, nlayer= args.nlayer, dropout= args.dropout)

    return Server(model=model, graph=global_graph, device=args.device)

if __name__ == '__main__':
    args = parse_args()
   
    split_seed = 123
    set_seed(args.seed)

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.method == "central":
        args.num_clients = 1
    
    # SETUP DATASET + SERVER/CLIENTS

    global_graph, subgraphs, num_classes, num_node_features = setup_dataset(args.dataset, num_clients=args.num_clients, partition_method= args.partition, seed = args.seed, split_seed=split_seed)
    args.num_classes = num_classes
    
    print(f"Subgraph construction from dataset {args.dataset} complete.")

    clients = init_clients(subgraphs=subgraphs, num_classes=num_classes, num_node_features=num_node_features, args=args)
    server = init_server(global_graph=global_graph, num_classes=num_classes, num_node_features=num_node_features, args=args)

    if args.skip_client: # for contribution testing
        if args.skip_client_idx >= args.num_clients:
             raise ValueError(f"skip client idx greater than num_cleints: {args.args.skip_client_idx} >= {args.num_clients:}")
        else:
            print(f"num of clients: {len(clients)}")
            clients = clients[0: args.skip_client_idx] 
            if args.skip_client_idx != len(clients) - 1:
                clients[args.skip_client_idx + 1:]
            print(f"client remove, new len: {len(clients)} ")

    # SELECT TRAINING METHOD
             
    incentives = False
    if args.method == "selftrain" or args.method == "central":
        metrics = selftrain(clients=clients, server=server, local_epoch=args.local_epoch)
    elif args.method == "fedavg":
        metrics = fedavg(clients=clients, server=server, communication_rounds=args.num_rounds, local_epoch=args.local_epoch, with_prototypes=False)
    elif args.method == "fedavg-proto":
        metrics = fedavg(clients=clients, server=server, communication_rounds=args.num_rounds, local_epoch=args.local_epoch, with_prototypes=True)
    elif args.method == "fairfed":
        incentives =  True
        metrics = fairfed(clients=clients, server=server, communication_rounds=args.num_rounds, local_epoch=args.local_epoch, with_prototypes=False)
    elif args.method == "fairfed-proto":
        incentives =  True
        metrics = fairfed(clients=clients, server=server, communication_rounds=args.num_rounds, local_epoch=args.local_epoch, with_prototypes=True)
    else:
        raise ValueError(f"Unknown training framework: {args.method}")

    # SAVE RESULTS
    
    server_stats, client_stats, client_incentives = collect_all_metrics(server, clients, num_classes, num_node_features, incentives)
    save_experiment_dataframes(args=args, server_stats=server_stats, client_stats=client_stats, client_incentives=client_incentives)



    
