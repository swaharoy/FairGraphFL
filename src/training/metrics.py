import pandas as pd
from client import Client
from server import Server

def collect_client_metrics(clients: list[Client]):
    frame = pd.DataFrame()

    for client in clients:
        _, acc = client.evaluate()
        frame.loc[client.id, 'test_acc'] = acc
        
        print("  > {} done.".format(client.id))

    return frame

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def display_metrics_table(frame):
    """
    Applies pandas styling and prints.
    """
    styled_frame = frame.style.apply(highlight_max)
    print(styled_frame.to_string())

def collect_and_print_client_metrics(clients):
    frame = collect_client_metrics(clients)
    display_metrics_table(frame)

    return frame

def collect_and_print_client_and_server_metrics(clients, server):
    frame = collect_and_print_client_metrics(clients)

    _, acc = server.eval_global_accuracy()
    print(f"global acc: {acc}")

    return frame


def collect_metrics(clients: list[Client], server: Server):
    diversity = server.client_diversity
    edge_loss = []
    personalized_acc = []
    
    for client in clients:
        edge_loss.append(client.subgraph.num_inter_edges)
        _, acc = client.evaluate()
        personalized_acc.append(acc)
    
    _, acc = server.eval_global_accuracy()