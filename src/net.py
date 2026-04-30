import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GCN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        # Match the pre/post structure or use basic layers
        self.pre = nn.Linear(nfeat, nhid)
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nhid, nhid))
        for _ in range(nlayer - 1):
            self.convs.append(GCNConv(nhid, nhid))
            
        self.post = nn.Linear(nhid, nclass)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pre-layer
        x = self.pre(x)
        
        # Convolution layers
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x2 = x  # Required for compatibility
        x1 = x  # Required for compatibility
        
        # Classification output
        x = self.post(x1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.log_softmax(x, dim=1)

        return x, x1, x2

    def loss(self, pred, label, mask=None):
        if mask is not None:
            return F.nll_loss(pred[mask], label[mask])
        return F.nll_loss(pred, label)