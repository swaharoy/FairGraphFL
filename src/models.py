import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class serverGIN(torch.nn.Module):
    """
    A simplified Graph Isomorphism Network (GIN) for the central server.
    
    This acts as a structural template to hold the aggregated global weights 
    for the convolutional layers.

    Args:
        nlayer (int): Number of GINConv layers.
        nhid (int): Dimensionality of hidden units.
    """
    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))


class GIN(torch.nn.Module):
    """
    Standard Graph Isomorphism Network (GIN) for local client training.

    Args:
        nfeat (int): Number of input node features.
        nhid (int): Dimensionality of hidden units.
        nclass (int): Number of output classes.
        nlayer (int): Number of GINConv layers.
        dropout (float): Dropout probability.
    """
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        """
        Performs a forward pass for Node Classification.

        Args:
            batched_data (torch_geometric.data.Data or Batch): The input graph data 
                containing node features (`x`), edge connectivity (`edge_index`), and 
                any edge attributes (`edge_attr`).

        Returns:
            tuple: A tuple of three tensors `(x, x1, x2)` representing different stages 
            of the node embeddings:
                - x (torch.Tensor): The final node-level class predictions (log-probabilities). 
                  Shape: [total_num_nodes, nclass]. Used for calculating classification loss.
                - x1 (torch.Tensor): The pre-prediction node embeddings. In this node 
                  classification setup, this is identical to `x2` and is returned to maintain 
                  API compatibility with existing graph-level federated evaluation functions. 
                  Shape: [total_num_nodes, nhid].
                - x2 (torch.Tensor): The raw node representations extracted directly from 
                  the core GNN message-passing layers. 
                  Shape: [total_num_nodes, nhid].
            # TODO: x2 can be removed layer, keeping for comptabilty rn
        """
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x2 = x
        x1 = x

        x = self.post(x1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)

        return x, x1, x2

    def loss(self, pred, label, mask=None):
        """
        Calculates the Negative Log-Likelihood loss.

        Args:
            pred (torch.Tensor): The model's log-probability predictions.
            label (torch.Tensor): The ground truth labels.
            mask (torch.Tensor, optional): Boolean mask to selectively calculate 
                loss on specific nodes (e.g., train_mask). Defaults to None.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        if mask is not None:
            return F.nll_loss(pred[mask].to(torch.float32), label[mask].view(-1,))
        return F.nll_loss(pred.to(torch.float32), label.view(-1,))






