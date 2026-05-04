"""
Prepare datasets for Subgraph Federated Learning (Node Classification)

    - Paritions globlal graphs into subgraphs varying levels of cross-subgraph edges.
"""

import torch
import torch_geometric
import pandas as pd
import os


from dataset.partition import partition_graph
from torch_geometric import datasets
import torch_geometric.transforms as T
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

from ogb.nodeproppred import PygNodePropPredDataset

torch.serialization.add_safe_globals([
    DataEdgeAttr,
    DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])

def create_global_test_mask(data, seed, test_ratio=0.10):
    """
    Creates a strict global test mask before partitioning.
    These nodes will be excluded from local training.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.num_nodes, generator=g)
    
    n_test = int(test_ratio * data.num_nodes)
    test_idx = perm[:n_test]
    
    global_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    global_test_mask[test_idx] = True
    
    data.global_test_mask = global_test_mask
    return data

def create_local_split(data, seed, train_ratio, val_ratio, exclude_mask):
    """
    Creates local train/val/test masks ONLY from nodes not in the exclude_mask.
    """
    num_nodes = data.num_nodes

    # map global test mask to the subgraph's local indices
    if hasattr(data, 'n_id'):
        global_indices = data.n_id #
    else:
        raise AttributeError("The subgraph does not have the 'n_id' attribute to map global node indices.")
    
    
    # find indices of nodes that are NOT in the global test set
    local_nodes_in_test = exclude_mask[global_indices] 
    eligible_nodes = torch.nonzero(~local_nodes_in_test, as_tuple=True)[0]
    num_eligible = len(eligible_nodes)
    
    # shuffle only the eligible nodes
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_eligible, generator=g)
    shuffled_eligible = eligible_nodes[perm]
    
    # calculate sizes (relative to the whole subgraph to hit target ratios)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    
    # ensure we don't try to assign more nodes than are eligible
    if n_train + n_val > num_eligible:
        n_train = int(num_eligible * 0.6) # Fallback to 60/20/20 of remaining
        n_val = int(num_eligible * 0.2)
        
    train_idx = shuffled_eligible[:n_train]
    val_idx = shuffled_eligible[n_train:n_train + n_val]
    test_idx = shuffled_eligible[n_train + n_val:] # Remaining eligible become local test
    
    # create local masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    return data

def compute_graph_stats(global_graph, subgraphs, undirected=True):
    """
    Computes structural statistics (nodes, edges) for the global and partitioned graphs.

    Args:
        global_graph (torch_geometric.data.Data): The original unpartitioned graph.
        subgraphs (list): A list of partitioned subgraphs (Data objects).
        undirected (bool): Whether to divide edge counts by 2 for undirected graphs.

    Returns:
        tuple: (global_df, subgraph_df) Pandas DataFrames containing the statistics.
    """
    # global graph stats
    num_nodes_global = global_graph.num_nodes
    num_edges_global = global_graph.edge_index.shape[1]

    if undirected:
        num_edges_global = num_edges_global // 2

    global_df = pd.DataFrame([{
        "num_nodes": num_nodes_global,
        "num_edges": num_edges_global,
    }])

    # subgraph stats
    subgraph_rows = []

    for i, sg in enumerate(subgraphs):

        num_nodes = sg.num_nodes

        # intra edges 
        num_intra_edges = sg.edge_index.shape[1]
        if undirected:
            num_intra_edges = num_intra_edges // 2

        # inter edges = what you stored
        num_inter_edges = getattr(sg, "num_inter_edges", None)

        subgraph_rows.append({
            "subgraph_id": i,
            "num_nodes": num_nodes,
            "num_intra_edges": num_intra_edges,
            "num_inter_edges": num_inter_edges,
        })

    subgraph_df = pd.DataFrame(subgraph_rows)

    return global_df, subgraph_df

def get_data(dataset):
    """
    Loads standard node classification datasets.

    Args:
        dataset (str): The name of the dataset to load.

    Returns:
        tuple: (data object, num_classes, num_node_features)
    """
    data_path = os.path.join("data", dataset)

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        ds = datasets.Planetoid(
            data_path,
            dataset,
            transform=T.NormalizeFeatures()
        )
        data = ds[0]
    elif dataset in ['Computers', 'Photo']:
        ds = datasets.Amazon(
            data_path,
            dataset,
            transform=T.NormalizeFeatures()
        )
        
        data = ds[0]

        # create empty masks (will fill later)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        ds = PygNodePropPredDataset(
            dataset,
            root=data_path,
            transform=T.ToUndirected()
        )
        
        data = ds[0]

        # fix label shape
        data.y = data.y.view(-1)

        # create masks
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
   
    return data, ds.num_classes, ds.num_node_features

def setup_dataset(dataset_name, num_clients, partition_method, seed, split_seed):
    """
    Full pipeline to load, split, and partition datasets for Subgraph Federated Learning.

    Args:
        dataset_name (str): Name of the dataset.
        num_clients (int): Number of clients to partition the graph for.
        partition_method (str): Algorithm for clustering/partitioning.
        seed (int): Seed for the partitioner.
        split_seed (int): Seed for generating train/val/test masks.

    Returns:
        tuple: (subgraphs list, global stats DF, client stats DF, num_classes, num_node_features)
    """
    global_graph, num_classes, num_node_features  = get_data(dataset_name)

    global_graph = create_global_test_mask(global_graph, seed=split_seed, test_ratio=0.1)  

    if num_clients == 1:
        global_graph.num_inter_edges = 0
        subgraphs = [global_graph]
    else:
        subgraphs = partition_graph(global_graph, num_subgraphs=num_clients, method=partition_method, seed = seed)

    # determine split ratios based on dataset
    if dataset_name == 'ogbn-arxiv':
        train_ratio = 0.05
        val_ratio = 0.475
    else:
        train_ratio = 0.20
        val_ratio = 0.40

    # apply the train/val/test splits PER SUBGRAPH
    for i in range(len(subgraphs)):
        # We add 'i' to the split_seed so that if two subgraphs happen to 
        # have the exact same number of nodes, they get differently randomized masks
        subgraphs[i] = create_local_split(
            subgraphs[i], 
            seed=(split_seed + i), 
            train_ratio=train_ratio, 
            val_ratio=val_ratio,
            exclude_mask=global_graph.global_test_mask
        )

    global_stats, client_stats = compute_graph_stats(global_graph, subgraphs)

    return global_graph, subgraphs, global_stats, client_stats, num_classes, num_node_features 


