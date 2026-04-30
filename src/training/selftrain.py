from client import Client
from server import Server

def selftrain(clients: list[Client], server: Server, local_epoch):
    """
    Executes isolated local training for all clients to establish a baseline.

    In this setup, clients download the initial global weights from the server 
    to ensure a fair starting point, but they do NOT collaborate or upload 
    their weights back to the server.

    Args:
        clients (list[Client]): A list of initialized Client objects.
        server (Server): The central server holding the initial global model.
        local_epoch (int): The number of epochs each client should train locally.

    Returns:
        dict: A dictionary mapping each client ID to a list containing their 
              final training accuracy, validation accuracy, and test accuracy.
    """
    # for client in clients:
    #     client.download_from_server(server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.id] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.id))

    return allAccs