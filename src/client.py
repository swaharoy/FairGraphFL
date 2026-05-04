import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans
import networkx as nx



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
        train_size (int) Number of nodes in client's train set.
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
        self.train_size = subgraph.train_mask.sum().item()
        
        self.optimizer = optimizer

        self.gconvNames = None # conv layer param names from server
        self.W = {name: param for name, param in self.model.named_parameters()}  # points to named_parameters()

        self.train_stats = {}

        self.reputation = [] # agent value at each c_round
        self.payoff = [] # agent payoff at each c_round

    def construct_motifs(self):
        """
        Constructs motifs (edges and small cycles) for a SINGLE subgraph partition.
        Ranks them by frequency and stores the node indices that form them.
        """
        self.motif_count = {}  # Tracks frequency of each motif
        self.motif_idx = {}    # Maps motif_key -> list of node tuples (e.g., [(0, 1), (5, 8)])
        self.prototype = {}    # Will hold the prototype embeddings {motif_key: embedding}
        
        # target the client's single assigned subgraph
        graph = self.subgraph  
        
        # determine "Node Types" using K-Means clustering for multi-hot features
        label = _assign_node_types_kmeans(graph.x, num_clusters=self.args.num_classes, random_state=self.args.seed)
        
        # convert to NetworkX
        if graph.edge_attr is not None:
            graph_net = to_networkx(graph, to_undirected=True, edge_attrs=["edge_attr"])
        else:
            graph_net = to_networkx(graph, to_undirected=True)
            
        # find rings (cycles of length <= 4)
        mcb = nx.cycle_basis(graph_net)
        mcb_tuple = [tuple(ele) for ele in mcb if len(ele) <= 4]
        
        # find edges not in any ring
        edges = []
        for e in graph_net.edges():
            count = sum(1 for c in mcb_tuple if e[0] in set(c) and e[1] in set(c))
            if count == 0:
                edges.append(e)
        edges = list(set(edges))
        
        # extract edge motifs
        for e in edges:
            weight = 1  # default weight unless using edge_attr
            
            # canonicalize edge (smaller label first) so (A, B) matches (B, A)
            l1, l2 = label[e[0]].item(), label[e[1]].item()
            c = (l1, l2) if l1 <= l2 else (l2, l1)
            motif_key = (c, weight)

            if motif_key not in self.motif_count:
                self.motif_count[motif_key] = 0
                self.motif_idx[motif_key] = []
                
            self.motif_count[motif_key] += 1
            
            # store exact node indices forming this motif
            self.motif_idx[motif_key].append((e[0], e[1])) 
            
        # extract ring motifs
        for m in mcb_tuple:
            # check if custom weight function exists, else default to 1s
            if hasattr(self, 'find_ring_weights'):
                weight = tuple(self.find_ring_weights(m, graph_net))
            else:
                weight = tuple([1] * len(m))
                
            # get labels for nodes in the ring
            ring_labels = [label[node].item() for node in m]
            
            # canonicalize ring (sort labels to ensure structural matching)
            c = tuple(sorted(ring_labels))
            motif_key = (c, weight)
            
            if motif_key not in self.motif_count:
                self.motif_count[motif_key] = 0
                self.motif_idx[motif_key] = []
                
            self.motif_count[motif_key] += 1
            
            # store exact node indices of the cycle
            self.motif_idx[motif_key].append(m) 
            
        # rank by frequeny (most to least)
        sorted_motifs = sorted(self.motif_count.items(), key=lambda x: x[1], reverse=True)
        
        # keep only the top motifs based on args.beta threshold
        num_to_keep = max(1, int(len(sorted_motifs) * self.args.beta))
        top_motifs = dict(sorted_motifs[:num_to_keep])
        
        # filter dictionaries to only keep the top motifs
        self.motif_count = top_motifs
        self.motif_idx = {k: v for k, v in self.motif_idx.items() if k in top_motifs}
        
        # initialize empty prototypes ready for embeddings
        for key in self.motif_count.keys():
            self.prototype[key] = []
            
        print(f'Client {self.id} constructed {len(self.prototype.keys())} motifs from its subgraph.')
        self.num_of_motifs = len(self.prototype.keys())

    def prototype_update(self):
        """
        Generates and normalizes structural-semantic prototypes for the client.
        
        This function implements the mapping of subgraph structures (motifs) to their 
        corresponding high-level feature representations (embeddings). It aggregates 
        the local 'structural identity' of the client into a set of normalized vectors 
        that are sent to the server to calculate reputation and similarity scores.

        Attributes updated:
            self.prototype (dict): A dictionary mapping motif keys to a single, 
                normalized 1D tensor representing the 'average' node embedding for 
                that specific structure.
        """
        # prepare the subgraph and model
        self.subgraph = self.subgraph.to(self.args.device)
        self.model.concat = False
        
        # pass the entire subgraph through the GNN to get NODE embeddings
        _, x1, _ = self.model(self.subgraph)
        node_embeddings = x1.data 
        
        # clear the prototype dictionary just in case it holds stale data from a previous epoch
        self.prototype = {}
        
        # map node embeddings to their respective motifs
        for key, instance_list in self.motif_idx.items():
            # extract all unique node IDs
            unique_nodes = set()
            for motif_instance in instance_list:
                # .update() automatically unpacks the tuple and adds its elements to the set
                unique_nodes.update(motif_instance) 
            
            # convert back to a list for PyTorch indexing
            indices = list(unique_nodes)
            
            # extract the embeddings of the specific nodes involved in this motif
            involved_nodes_emb = node_embeddings[indices]
            
            # average embedding of these specific nodes
            self.prototype[key] = involved_nodes_emb.mean(dim=0)
            
        # normalize the prototypes
        for key in self.prototype.keys():
            norm_value = torch.norm(self.prototype[key])
            # Use 1e-8 instead of ones_like to cleanly prevent division by zero
            self.prototype[key] /= torch.max(norm_value, torch.tensor(1e-8, device=self.args.device))

        
    def train_with_prototypes(self, server):
        """
        Trains the local model using both the standard classification loss 
        and the structural prototype alignment (MSE) loss (Equation 14).
        
        Args:
            server (Server): The central server holding the global prototypes.
            local_epoch (int, optional): Overrides args.local_epoch if provided.
        """

        local_epoch = self.args.local_epoch

        self.model.concat = True  # TODO: verify purpose
        loss_mse = torch.nn.MSELoss()
        data = self.subgraph.to(self.device)

        for epoch in range(local_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            # forward pass (process entire subgraph)
            pred, node_embeddings, _ = self.model(data)

            # task loss (L_GNN)
            loss1 = self.model.loss(pred, data.y, data.train_mask)

            # prototype alignment loss (L_proto)
            loss2 = 0.0
            num_motifs = len(self.motif_idx.keys())
            
            if num_motifs > 0:
                for key, instance_list in self.motif_idx.items():
                    # extract unique nodes forming this motif
                    unique_nodes = set()
                    for motif_instance in instance_list:
                        unique_nodes.update(motif_instance)
                    indices = list(unique_nodes)

                    # calculate current local average embedding for this motif
                    involved_nodes_emb = node_embeddings[indices]
                    current_local_proto = involved_nodes_emb.mean(dim=0)

                    # fetch the global prototype from the server
                    global_proto = server.global_prototype[key].to(self.device)

                    # calculate MSE distance
                    loss2 += loss_mse(current_local_proto, global_proto)
                
                loss2 = loss2 / num_motifs  # avg over all local motifs

            # total Loss (Equation 14)
            loss = loss1 + (self.args.lamb * loss2)
            
            loss.backward()
            self.optimizer.step()            


    def download_weights_from_server(self, server):
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



def _assign_node_types_kmeans(x: torch.Tensor, num_clusters: int, random_state: int) -> torch.Tensor:
    """
    Assigns discrete structural 'types' to nodes based on their continuous or multi-hot features.
    
    This function uses K-Means clustering to group high-dimensional node features 
    (e.g., Bag-of-Words vectors, TF-IDF, or dense embeddings) into `num_clusters` distinct 
    semantic buckets. This many-to-one mapping forces collisions, which is strictly required 
    for frequent subgraph mining algorithms to discover recurring structural motifs.

    Args:
        x (torch.Tensor): The node feature matrix of shape [num_nodes, num_features].
        num_clusters (int): The number of distinct node types to create (K). 
                            Default is 7 (a good baseline for Cora's 7 classes).
        random_state (int): Seed for reproducible, deterministic clustering.

    Returns:
        torch.Tensor: A 1D tensor of shape [num_nodes] containing the integer 
                      node type (cluster ID) for each node.
    """
    # scikit-learn requires NumPy arrays, so we detach and move to CPU
    x_np = x.detach().cpu().numpy()
    
    # init K-Means (n_init=10 is explicitly set to suppress sklearn warnings)
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    
    # fit the clustering model and predict the cluster ID for every single node
    cluster_ids = kmeans.fit_predict(x_np)
    
    # convert the resulting IDs back to a PyTorch tensor and send to the original device
    labels = torch.tensor(cluster_ids, dtype=torch.long, device=x.device)
    
    return labels