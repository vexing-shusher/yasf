import torch
import numpy as np
from torch.nn import Module

class STDP(Module):
    # language=rst


    def __init__(
        self,
        tau_plus : float = 10,
        tau_minus : float = 10,
        A_plus : float = 0.01,
        A_minus : float = 0.01,
    ) -> None:
        
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus

    def x(self, dt): #pre-synaptic
        return torch.exp(-dt/self.tau_plus)

    def y(self, dt): #post-synaptic
        return torch.exp(dt/self.tau_minus)

    def __call__(self, 
                 input_times : torch.Tensor, 
                 output_times : torch.Tensor,
                )->torch.Tensor:
        
        assert input_times.device == output_times.device, "Input and output must be on the same device."
        
        delta_t = output_times.unsqueeze(0).repeat((len(input_times),1)) - input_times.unsqueeze(1).repeat((1,len(output_times)))
        
        sign = torch.sign(delta_t)
        
        mask = (sign==0)
        
        pos = self.A_plus * self.x(delta_t) * (1 + sign) * 0.5

        neg = self.A_minus * self.y(delta_t) * (1 - sign) * 0.5
        
        pos[mask] = 0
        neg[mask] = 0
        
        del sign
        del delta_t
        
        return pos-neg


class ncSTDP(Module):
    # language=rst


    def __init__(
        self,
        mu_plus : float = 26.700,
        mu_minus : float = -22.300,
        tau_plus : float = 9.300,
        tau_minus : float = -10.800,
        A_plus : float = 0.074,
        A_minus : float = 0.047,
        wmax : float = 1.,
    ) -> None:
        

        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus/wmax
        self.A_minus = A_minus/wmax

    def x(self, dt): #pre-synaptic
        return torch.ones(dt.shape).to(dt.device) + torch.tanh(-(dt - self.mu_plus) / self.tau_plus)

    def y(self, dt): #post-synaptic
        return torch.ones(dt.shape).to(dt.device) - torch.tanh((dt - self.mu_minus) / self.tau_minus)

    def __call__(self, 
                 input_times : torch.Tensor, 
                 output_times : torch.Tensor,
                 w : torch.Tensor,
                )->torch.Tensor:
        
        assert input_times.device == output_times.device, "Input and output must be on the same device."
        assert input_times.device == w.device, "Input times and weights must be on the same device."
        
        delta_t = output_times.unsqueeze(0).repeat((len(input_times),1)) - input_times.unsqueeze(1).repeat((1,len(output_times)))
        
        sign = torch.sign(delta_t)
        
        mask = (sign==0)
        
        pos = self.A_plus * w * self.x(delta_t) * (1 + sign) * 0.5

        neg = self.A_minus * w * self.y(delta_t) * (1 - sign) * 0.5
        
        pos[mask] = 0
        neg[mask] = 0
        
        del sign
        del delta_t
        
        return pos-neg
    
class ppxSTDP(Module):
    # language=rst


    def __init__(
        self,
        alpha_plus : float = 0.316,
        alpha_minus : float = 0.011,
        beta_plus : float = 2.213,
        beta_minus : float = -5.969,
        gamma_plus : float = 0.0318,
        gamma_minus : float = 0.146,
        wmax : float = 1.,
        wmin : float = 0.,
        t_norm : float = 10.,
    ) -> None:
        

        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.beta_plus = beta_plus
        self.beta_minus = beta_minus
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus
        
        self.wmax = wmax
        self.wmin = wmin
        
        self.t_norm = t_norm

    def A_plus(self, w): #pre-synaptic
        return self.alpha_plus * torch.exp(-self.beta_plus * (self.wmax - w) / (self.wmax - self.wmin))

    def A_minus(self, w): #post-synaptic
        return self.alpha_minus * torch.exp(-self.beta_minus * (w - self.wmin) / (self.wmax - self.wmin))

    def x(self, dt): #pre-synaptic
        return torch.abs(dt/self.t_norm)*torch.exp(-self.gamma_plus * torch.square((dt/self.t_norm)))

    def y(self, dt): #post-synaptic
        return torch.abs(dt/self.t_norm)*torch.exp(-self.gamma_minus * torch.square((dt/self.t_norm))) 

    def __call__(self, 
                 input_times : torch.Tensor, 
                 output_times : torch.Tensor,
                 w : torch.Tensor,
                )->torch.Tensor:
        
        assert input_times.device == output_times.device, "Input and output must be on the same device."
        assert input_times.device == w.device, "Input times and weights must be on the same device."
        
        delta_t = output_times.unsqueeze(0).repeat((len(input_times),1)) - input_times.unsqueeze(1).repeat((1,len(output_times)))
        
        sign = torch.sign(delta_t)
        
        mask = (sign==0)
        
        pos = self.A_plus(w) * self.x(delta_t) * (1 + sign) * 0.5

        neg = self.A_minus(w) * self.y(delta_t) * (1 - sign) * 0.5
        
        pos[mask] = 0
        neg[mask] = 0
        
        del sign
        del delta_t
        
        return pos-neg
