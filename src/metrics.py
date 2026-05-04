import pandas as pd
from client import Client
from server import Server


def compute_client_stats(clients: list[Client], client_diversity):
    """
    Computes statistics for each client and compiles them into a DataFrame.

    Args:
        clients (list[Client]): List of Client objects.
        client_diversity: list client diversity values.

    Returns:
        pd.DataFrame: DataFrame containing client statistics.
    """
    client_rows = []

    for i, client in enumerate(clients):
        sg = client.subgraph
        num_nodes = sg.num_nodes

        num_intra_edges = sg.edge_index.shape[1]
        num_intra_edges = num_intra_edges // 2

        num_inter_edges = getattr(sg, "num_inter_edges", None)

        train_size = sg.train_mask.sum().item()
        val_size = sg.val_mask.sum().item()
        test_size = sg.test_mask.sum().item()

        _, acc = client.evaluate()

        diversity = client_diversity[i] if client_diversity is not None and i < len(client_diversity) else None
        num_of_motifs = getattr(client, "num_of_motifs", None)

        client_rows.append({
            "client_id": client.id,
            "num_nodes": num_nodes,
            "num_intra_edges": num_intra_edges,
            "edge_loss": num_inter_edges,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "diversity": diversity,
            "num_of_motifs": num_of_motifs,
            "acc": acc
        })
    
    clients_df = pd.DataFrame(client_rows)

    return clients_df

def collect_client_incentives(clients: list[Client]):
    """
    Creates a DataFrame containing reputation and payoff values across communication rounds.

    Args:
        clients (list[Client]): List of Client objects.

    Returns:
        pd.DataFrame: A DataFrame where each row is a communication round 
                      and each column is a client ID.
    """
    client_ids = [c.id for c in clients]

    # each row is communication round, each col is client id
    columns = pd.MultiIndex.from_product(
                    [['reputation', 'payoff'], client_ids],
                    names=['metric', 'client_id']
                    )
    
    num_rounds = len(clients[0].payoff)

    # combine the data so each row represents a round
    combined_data = []
    for r in range(num_rounds):
        rep_r = [client.reputation[r + 1].item() for client in clients] # skip first idx in reputation (1/N)
        pay_r = [client.payoff[r].item() for client in clients]
        combined_data.append(rep_r + pay_r)
        
    # create the MultiIndex DataFrame
    df = pd.DataFrame(combined_data, columns=columns)
    df.index.name = 'communication_round'

    return df

def compute_server_stats(server: Server, num_classes, num_node_features):
    """
    Computes server statistics and compiles them into a DataFrame.

    Args:
        server (Server): The Server object.
        num_classes (int): Number of classes.
        num_node_features (int): Number of node features.

    Returns:
        pd.DataFrame: DataFrame containing server statistics.
    """    
    _, acc = server.eval_global_accuracy()

    global_graph = server.graph
    num_nodes_global = global_graph.num_nodes

    num_edges_global = global_graph.edge_index.shape[1]
    num_edges_global = num_edges_global // 2

    test_size = global_graph.global_test_mask.sum().item()

    df = pd.DataFrame([{
        "num_nodes": num_nodes_global,
        "num_edges": num_edges_global,
        "test_size": test_size,
        "num_classes": num_classes,
        "num_node_features": num_node_features,
        "acc": acc
    }])

    df = pd.DataFrame(df)

    return df


def collect_all_metrics(server: Server, clients: list[Client], num_classes, num_node_features, incentives=False):
    """
    Collects all metrics from both the server and clients.

    Args:
        server (Server): The Server object.
        clients (list[Client]): List of Client objects.
        num_classes (int): Number of classes.
        num_node_features (int): Number of node features.
        incentives (bool): Whether to collect client incentives.

    Returns:
        tuple: (server_stats, client_stats, client_incentives) DataFrames.
    """
   
    server_stats = compute_server_stats(server, num_classes, num_node_features)
    client_stats = compute_client_stats(clients, server.client_diversity)
    client_incentives = collect_client_incentives(clients) if incentives else None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("--- SERVER STATS --- ")
    print(server_stats)
    print("\n")
    print("--- CLIENT STATS --- ")
    print(client_stats)
    print("\n")
    print("--- INCENTIVES --- ")
    print(client_incentives.iloc[::40])

    return server_stats, client_stats, client_incentives
