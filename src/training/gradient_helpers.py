import torch
from torch.nn.utils import parameters_to_vector

def flatten(tensor_list):
    """
    Flattens a list of tensors into a single 1D vector.
    """
    return parameters_to_vector(tensor_list)

def unflatten(flattened_tensor, reference_tensors):
    """
    Unflattens a 1D vector back into a list of tensors matching the shapes 
    of the reference_tensors.
    
    Args:
        flattened_tensor (torch.Tensor): The 1D vector.
        reference_tensors (list[torch.Tensor]): List of tensors whose shapes define the target structure.
        
    Returns:
        list[torch.Tensor]: A list of unflattened tensors.
    """
    grad_update = []
    pointer = 0
    
    for param in reference_tensors:
        num_params = param.numel()  # Total number of elements in this tensor
        
        # Extract the segment and reshape it to match the original tensor
        shape = param.shape
        param_slice = flattened_tensor[pointer : pointer + num_params].view(shape)
        
        grad_update.append(param_slice)
        pointer += num_params
        
    return grad_update