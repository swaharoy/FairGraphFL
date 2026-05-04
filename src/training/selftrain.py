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
    """
    for client in clients:
        client.download_weights_from_server(server)

    for client in clients:
        client.local_train(local_epoch)
