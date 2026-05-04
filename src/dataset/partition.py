"""
    Adapted from:

    @misc{aliakbari2025decoupledsubgraphfederatedlearning,
      title={Decoupled Subgraph Federated Learning}, 
      author={Javad Aliakbari and Johan Östman and Alexandre Graell i Amat},
      year={2025},
      eprint={2402.19163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.19163}, 
}
"""

from itertools import chain
from collections import defaultdict

import torch
import numpy as np
import networkx as nx
from sklearn.cluster import k_means
from torch_geometric.data import Data


def partition_graph(graph, num_subgraphs, method, seed, delta=20):
    if method == "louvain":
        subgraph_node_ids = louvain_cut(
            graph.edge_index, graph.num_nodes, num_subgraphs, delta, seed
        )
    elif method == "random":
        subgraph_node_ids = random_assign(graph.num_nodes, num_subgraphs)
    elif method == "kmeans":
        subgraph_node_ids = kmeans_cut(graph.x, num_subgraphs, delta, seed)
    elif method == "metis":
        subgraph_node_ids = metis_cut(graph.edge_index, graph.num_nodes, num_subgraphs)

    return create_subgraphs(graph, subgraph_node_ids)


### COMMUNITY DETECTION
def find_community(edge_index, num_nodes, seed):
    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G, seed=seed)

    community_nodes = torch.tensor(list(chain.from_iterable(community)))
    node_ids = torch.arange(num_nodes)

    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[community_nodes] = False
    isolated_nodes = node_ids[mask]

    community.append(isolated_nodes)
    community = {ind: list(c) for ind, c in enumerate(community)}

    return community

def create_community_groups(community_map) -> dict:
    community_groups = defaultdict(list)

    for node_id, community in enumerate(community_map):
        community_groups[community].append(node_id)

    return community_groups


### GROUP BALANCING
def make_groups_smaller_than_max(community_groups, group_len_max) -> dict:
    ind = 0
    while ind < len(community_groups):
        if len(community_groups[ind]) > group_len_max:
            l1, l2 = (
                community_groups[ind][:group_len_max],
                community_groups[ind][group_len_max:],
            )

            community_groups[ind] = l1
            community_groups[len(community_groups)] = l2

        ind += 1

    return community_groups

def assign_nodes_to_subgraphs(community_groups, num_nodes, num_subgraphs, delta):
    max_subgraph_nodes = num_nodes // num_subgraphs
    subgraph_node_ids = {subgraph_id: [] for subgraph_id in range(num_subgraphs)}
    
    current_ind = 0
    counter = 0

    for community in community_groups.keys():
        while (
            len(subgraph_node_ids[current_ind]) + len(community_groups[community])
            > max_subgraph_nodes + delta
            or len(subgraph_node_ids[current_ind]) >= max_subgraph_nodes
        ):
            current_ind += 1
            if current_ind == num_subgraphs:
                current_ind = 0

            counter += 1
            if counter == num_subgraphs:
                current_ind = np.argmin([len(s) for s in subgraph_node_ids.values()])
                break

        subgraph_node_ids[current_ind] += community_groups[community]
        counter = 0

    assert sum([len(s) for s in subgraph_node_ids.values()]) == num_nodes

    return subgraph_node_ids


### SUBGRAPHS
def build_subgraph(graph, node_ids):
    if not isinstance(node_ids, torch.Tensor):
        node_ids = torch.tensor(node_ids, dtype=torch.long)

    # create mask for nodes
    node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    node_mask[node_ids] = True

    row, col = graph.edge_index
    
    # inter edges: exactly one endpoint in subgraph (edge cut)
    inter_mask = node_mask[row] ^ node_mask[col]

    # filter edges, keep ONLY intra-subgraph edges
    intra_mask = node_mask[row] & node_mask[col]
    edge_index = graph.edge_index[:, intra_mask]

    # remap global to local indices
    global_to_local = -torch.ones(graph.num_nodes, dtype=torch.long)
    global_to_local[node_ids] = torch.arange(len(node_ids))

    edge_index = global_to_local[edge_index]

    # slice features & labels 
    x = graph.x[node_ids] if graph.x is not None else None
    y = graph.y[node_ids] if graph.y is not None else None

    # masks (optional)
    def slice_mask(mask):
        return mask[node_ids] if mask is not None else None

    train_mask = slice_mask(getattr(graph, "train_mask", None))
    val_mask = slice_mask(getattr(graph, "val_mask", None))
    test_mask = slice_mask(getattr(graph, "test_mask", None))

    subgraph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    subgraph.num_inter_edges = inter_mask.sum().item() // 2
    subgraph.n_id = node_ids

    return subgraph

def create_subgraphs(data, subgraph_node_ids):
    """
    CHANGED:
    - now returns list of PyG Data objects
    - no custom Graph class
    """
    subgraphs = []
    for _, node_ids in subgraph_node_ids.items():
        subgraphs.append(build_subgraph(data, node_ids))
    return subgraphs


### PARTITION METHODS
def random_assign(num_nodes, num_subgraphs): #TODO: determine if we want balanced size?
    subgraph_id = np.random.choice(num_subgraphs, num_nodes, replace=True)
    subgraph_node_ids = {
        value: torch.tensor(
            np.where(subgraph_id == value)[0], dtype=torch.int64
        )
        for value in range(num_subgraphs)
    }

    return subgraph_node_ids

def kmeans_cut(X, num_subgraphs, delta, seed):
    num_nodes = X.shape[0]
    _, subgraph_id, _ = k_means(X.cpu(), num_subgraphs, n_init="auto", random_state=seed)
    community_groups = create_community_groups(subgraph_id)

    group_len_max = num_nodes // num_subgraphs + delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs, delta
    )

    return subgraph_node_ids

def metis_cut(edge_index, num_nodes, num_subgraphs):
    import metis # TODO: ensure corrects packages installed

    edges = edge_index.T.tolist()
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edges)
    (edgecuts, community_map) = metis.part_graph(nx_graph, num_subgraphs)
    community_groups = create_community_groups(community_map=community_map)

    return community_groups

def louvain_cut(edge_index, num_nodes, num_subgraphs, delta, seed):
    community_groups = find_community(edge_index, num_nodes, seed)

    group_len_max = num_nodes // num_subgraphs + delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs, delta
    )

    return subgraph_node_ids



