class Server():
    """
    Central server module for Federated Learning.

    The server holds the global master model. In baseline training (selftrain), 
    it simply acts as a common initialization point. In full federated learning, 
    it handles the aggregation of client weights.

    Attributes:
        model (torch.nn.Module): The global graph neural network model.
        W (dict): A dictionary referencing the model's named parameters.
    """
    def __init__(self, model, device):
        """
        Initializes the Server.

        Args:
            model (torch.nn.Module): The global GNN model instance.
            device (torch.device): The device (CPU or CUDA) to host the model on.
        """
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}  # points to named_parameters()
    
