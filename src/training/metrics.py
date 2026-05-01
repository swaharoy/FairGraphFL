import pandas as pd
from client import Client

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