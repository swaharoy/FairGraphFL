"""
Prepare datasets for Subgraph Federated Learning (Node Classification)

    - Paritions globlal graphs into subgraphs varying levels of cross-subgraph edges.
"""

import torch
import torch_geometric
import pandas as pd
import os
import numpy as np


from data.partition import partition_graph
from torch_geometric import datasets
import torch_geometric.transforms as T
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

from ogb.nodeproppred import PygNodePropPredDataset

torch.serialization.add_safe_globals([
    DataEdgeAttr,
    DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])


def split_train_val_test(data, seed, train_ratio=0.2, val_ratio=0.35):
    """
    Creates randomized train, validation, and test masks for a global graph.

    Args:
        data (torch_geometric.data.Data): The input graph data.
        seed (int): Random seed for reproducibility.
        train_ratio (float): Proportion of nodes to use for training.
        val_ratio (float): Proportion of nodes to use for validation.

    Returns:
        torch_geometric.data.Data: The updated data object with masks attached.
    """

    num_nodes = data.num_nodes

    # reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # random permutation of all nodes
    perm = torch.randperm(num_nodes, generator=g)

    # compute sizes
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    # split indices
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # assign
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # attach to data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

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

    global_graph = split_train_val_test(global_graph, split_seed)

    if num_clients == 1:
        global_graph.num_inter_edges = 0
        subgraphs = [global_graph]
    else:
        subgraphs = partition_graph(global_graph, num_subgraphs=num_clients, method=partition_method, seed = seed)

    global_stats, client_stats = compute_graph_stats(global_graph, subgraphs)

    return subgraphs, global_stats, client_stats, num_classes, num_node_features 


