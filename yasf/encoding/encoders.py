import torch
import numpy as np

class AbstractEncoder(object):
    
    def __init__(self, 
                 time : float, 
                 dt : float,  
                 device : torch.device = torch.device("cpu"),
                )->None:
        
        self.time = time
        self.dt = dt
        self.device = device
        
    def __call__(self,)->None:
        pass
    

class PoissonEncoder(AbstractEncoder):
    
    """
    Functions similarly to the poisson encoding in BindsNET.
    """
    
    def __init__(self, 
                 time : float, 
                 dt : float, 
                 intensity : float, 
                 device : torch.device = torch.device("cpu"),
                )->None:
        
        super().__init__(time = time,
                         dt = dt,
                         device = device,)
        
        
        self.intensity = intensity
        
    def __call__(self, data : torch.Tensor)->torch.Tensor:
        
        assert (data>=0).all(), "Inputs must be non-negative."
        
        datum = data * self.intensity
        
        shape, size = datum.shape, datum.numel()
        datum = datum.flatten()
        time = int(self.time / self.dt)

        rate = torch.zeros(size, device=self.device)
        rate[datum != 0] = 1 / datum[datum != 0] * (1000 / self.dt)

        dist = torch.distributions.Poisson(rate=rate, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([time + 1]))
        intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

        times = torch.cumsum(intervals, dim=0).long()
        times[times >= time + 1] = 0
        
        spikes = torch.zeros(time + 1, size, device=self.device).byte()
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[1:]
        
        return spikes.view(time, *shape)