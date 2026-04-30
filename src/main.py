import os
import argparse
import random
import torch
import numpy as np
from pathlib import Path
import copy

from data import setup_dataset
from src.models import GIN, ogbGIN, serverGIN
from archive_src.client import Client_GC
from archive_src.server import Server

from src.training.selftrain import selftrain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='IMDB-BINARY')
    parser.add_argument('--dataset', help='specify the  datasets',
                        type=str, default='Cora')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--partition', help='subgraph partitioning method',
                        type=str, default='random')
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.03)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.06)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.85)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--disable_dp', type=bool, default=False)

    parser.add_argument('--training', help='FL training framework',
                    type=str, default='fedavg')

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_stats_outpath(args, is_global):
    dir_path = os.path.join(args.outbase, "initPartitioningImpl")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {dir_path}")

    prefix = "global_" if is_global else "subgraph_"
    filename = f"{args.dataset}_{args.partition}_{prefix}.csv"

    return os.path.join(dir_path, filename)

def init_clients(subgraphs, args):
    idx_clients = {}
    clients = []

    for idx, subgraph in enumerate(subgraphs):
        idx_clients[idx] = subgraph

        if args.data_group == 'ogb':
            model = ogbGIN(num_graph_labels, args.hidden, args.nlayer, args.dropout)
        else:
            model = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(model, idx, subgraph, train_size, graphs_train, dataloaders, optimizer, args))

    return clients

def init_server(dataset_name):
    if args.data_group == 'ogb':
        model = ogbGIN(num_graph_labels, args.hidden, args.nlayer, args.dropout)
    else:
        model = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
  
    return Server(model, args.device)

if __name__ == '__main__':
    args = parse_args()
   
    split_seed = 123
    set_seed(args.seed)

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    subgraphs, global_stats, subgraph_stats = setup_dataset.setup_datasets(args.dataset, num_clients=args.num_clients, partition_method= args.partition, seed = args.seed, split_seed=split_seed)

    print(f"Subgraph construction from dataset {args.dataset} complete.")

    outf_global = create_stats_outpath(args, is_global = True)
    outf_subgraph = create_stats_outpath(args, is_global = False)
    global_stats.to_csv(outf_global)
    subgraph_stats.to_csv(outf_subgraph)
    print(f"Wrote to {outf_global} and {outf_subgraph}")

    clients = init_clients(subgraphs, args)
    server = init_server(args.dataset)

    if args.training == "selftrain":
        selftrain(clients, server, args.local_epoch)
    elif args.training == "fedavg":
        pass
    elif args.training == "fairfedmotif":
        pass
    else:
        raise ValueError(f"Unknown training framework: {args.training}")

        


    
