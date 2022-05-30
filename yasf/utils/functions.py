import torch

def normalize(weights : torch.Tensor, norm : float) -> torch.Tensor:

    """
    Normalize weights so each target neuron has sum of connection weights equal to
    ``norm``.
    """
    w_abs_sum = weights.abs().sum(0).unsqueeze(0)
    w_abs_sum[w_abs_sum == 0] = 1.0
    
    
    return weights * norm / w_abs_sum
