from client import Client
from server import Server
from training.metrics import collect_and_print_client_metrics


def fedavg(clients: list[Client], server: Server, communication_rounds, local_epoch, frac=1.0):
    """
    Executes the FedAvg training loop.

    Args:
        clients (list[Client]): A list of initialized client objects participating in the training.
        server (Server): The central server coordinating the global model.
        communication_rounds (int): The total number of federated training rounds.
        local_epoch (int): The number of local training epochs each client performs per round.
        frac (float, optional): The fraction of clients randomly sampled each round. Defaults to 1.0.

    Returns:
        pandas.DataFrame: A dataframe containing the final evaluation metrics for all clients.
    """
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.rand_sample_clients(clients, frac)

        for client in selected_clients:
            client.download_from_server(server)
        
        for client in selected_clients:
            client.local_train(local_epoch)
        
        server.aggregate_weights(selected_clients)

    # sync all clients w server
    for client in clients:
        print(f"Client {client.id} train size: {client.train_size}")
        client.download_from_server(server)

        

    metrics_df = collect_and_print_client_metrics(clients)
    return metrics_df
