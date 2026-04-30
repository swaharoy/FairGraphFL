import torch

class Client():
    """
    Client module for local Federated Learning on subgraphs.

    Handles full-batch node classification training, evaluation, and weight 
    synchronization for a single localized subgraph.

    Attributes:
        id (int/str): Unique identifier for the client.
        args (Namespace): Configuration arguments (must contain `.device`).
        device (torch.device): The computation device.
        model (torch.nn.Module): The local GNN model.
        subgraph (torch_geometric.data.Data): The client's local graph data.
        optimizer (torch.optim.Optimizer): The optimizer for local training.
        W (dict): A dictionary of the local model's named parameters.
        gconvNames (list): Keys of the global convolution layers from the server.
        train_stats (dict): Tracks loss and accuracy metrics across epochs.
    """
    def __init__(self, client_id, model, subgraph, optimizer, args):
        self.id = client_id
        self.args = args
        self.device = args.device

        self.model = model.to(self.device)

        self.subgraph = subgraph
        
        self.optimizer = optimizer

        self.gconvNames = None # conv layer param names from server
        self.W = {name: param for name, param in self.model.named_parameters()}  # points to named_parameters()

        self.train_stats = {}


    def download_from_server(self, server):
        """
        Synchronizes the client's local weights with the server's global weights.

        Args:
            server (Server): The central server instance holding global weights.
        """
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()
    
    def local_train(self, local_epoch):
        """
        Performs full-batch local training on the client's subgraph.

        Args:
            local_epoch (int): The number of iterations to train over the subgraph.
        """
        losses_train, accs_train = [], []
        losses_val, accs_val = [], []
        losses_test, accs_test = [], []

        data = self.subgraph.to(self.device)

        for epoch in range(local_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            # process whole subgraph at once
            pred, _, _ = self.model(data)

            # calc loss on training nodes
            loss = self.model.loss(pred, data.y, data.train_mask)
            loss.backward()
            self.optimizer.step()

            # calc metrics
            correct_predictions = pred[data.train_mask].max(dim=1)[1].eq(data.y[data.train_mask]).sum().item()
            num_train_nodes = data.train_mask.sum().item()
            
            # prevent division by zero if subgraph has 0 training nodes
            acc = correct_predictions / max(num_train_nodes, 1)

            # eval on validation and test sets
            loss_v, acc_v = self._eval('val')
            loss_tt, acc_tt = self._eval('test')

            losses_train.append(loss.item())
            accs_train.append(acc)
            losses_val.append(loss_v)
            accs_val.append(acc_v)
            losses_test.append(loss_tt)
            accs_test.append(acc_tt)

        
        self.train_stats =  {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}

    def evaluate(self):
        """
        Evaluates the current local model purely on the test split.

        Returns:
            tuple: (test_loss, test_accuracy)
        """
        return self._eval('test')
    
    def _eval(self, split):
        """
        Internal evaluation function handling specific data splits.

        Args:
            split (str): Indicates which mask to apply ('train', 'val', or 'test').

        Returns:
            tuple: The calculated loss and accuracy (float, float) for the split.
        """
        self.model.eval()

        data = self.subgraph.to(self.device)
      
        with torch.no_grad():
            pred, _, _ = self.model(data)
            label = data.y

            if split == 'val':
                mask = data.val_mask
            elif split == 'test':
                mask = data.test_mask
            else:
                mask = data.train_mask

            loss = self.model.loss(pred, label, mask)

            correct_predictions = pred[mask].max(dim=1)[1].eq(label[mask]).sum().item()
            num_eval_nodes = mask.sum().item()
           
            if num_eval_nodes > 0:
                final_loss = loss.item()
                final_acc = correct_predictions / num_eval_nodes
            else:
                final_loss = 0.0
                final_acc = 0.0

        return final_loss, final_acc
