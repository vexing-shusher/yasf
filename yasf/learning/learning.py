import torch
import numpy as np
from torch.nn import Module

class STDP(Module):
    
    def __init__(self, 
                 tau_plus : float = 10., 
                 tau_minus : float = 10., 
                 A_plus : float = 0.1, 
                 A_minus : float = 0.1,
                ):
        
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        
    def __call__(self, 
                 input_times : torch.Tensor, 
                 output_times : torch.Tensor,)->torch.Tensor:
        
        assert input_times.device == output_times.device, "Input and output must be on the same device."
        
        input_trace = torch.exp(-input_times / self.tau_plus).view(-1,1)
        output_trace = torch.exp(-output_times / self.tau_minus).view(-1,1)

        dw_plus = self.A_plus * input_trace
        dw_minus = self.A_minus * output_trace
        
        dw_plus = dw_plus.repeat(1, output_trace.shape[0])
        dw_minus = dw_minus.repeat(1, input_trace.shape[0]).T
        
        dw = dw_plus - dw_minus
        
        return dw