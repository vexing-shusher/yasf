import torch
import numpy as np
import warnings

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
    
class GRFEncoder(AbstractEncoder):
    """
        Class for receptive fields data conversion.
    """
    def __init__(self, 
                 sigma2 : float, 
                 max_x : float, 
                 n_fields : int, 
                 dt : float, 
                 time : float,
                 scale : float = 1.0,
                 device = torch.device("cpu"),
                 reverse : bool = False, 
                 no_last : bool = False)->None:
        
        super().__init__(time = time,
                         dt = dt,
                         device = device,)
        
        self.sigma2 = torch.as_tensor(sigma2, dtype = torch.float32, device = device)
        self.max_x = torch.as_tensor(max_x, dtype = torch.float32, device = device)
        self.n_fields = n_fields
        self.scale = torch.as_tensor(scale, dtype = torch.float32, device = device)
        
        #to speed up calculations, transform function choice is performed during __init__
        
        if reverse and no_last:
            self.transform_samples = self._reverse_no_last
            
        elif reverse and not no_last:
            self.transform_samples = self._reverse
            
        elif not reverse and no_last:
            self.transform_samples = self._regular_no_last
            
        elif not reverse and not no_last:
            self.transform_samples = self._regular
        
        #rounding is performed to the first significant decimal place of dt
        
        self.k_round = 0
        while self.dt * (10 ** self.k_round) < 1:
            self.k_round += 1
            

    def get_gaussian(self, x : torch.Tensor, sigma2 : torch.Tensor, mu : torch.Tensor)->torch.Tensor:
        return (1 / torch.sqrt(2 * sigma2 * np.pi)) * torch.exp(- (x - mu) ** 2 / (2 * sigma2)).to(self.device)
    
    def get_spikes(self, x : torch.Tensor, y : torch.Tensor, n_classes : int)->torch.Tensor:
        
        spike_times = (x/self.dt).long()
        
        assert spike_times.max() < int(self.time/self.dt), f"Maximal spike arrival time is {spike_times.max() * self.dt} ms. Please, increase time."
        
        if spike_times.max() * 3 < int(self.time/self.dt):
            warnings.warn(f"Simulation time is at least {int(self.time/self.dt/spike_times.max())} times larger than last spike time. Consider lowering it. Last spike time is {spike_times.max() * self.dt} ms.")

        #generate spike patterns
        
        spikes = torch.nn.functional.one_hot(spike_times, num_classes = int(self.time/self.dt))
        
        spikes = spikes.squeeze().permute(0,2,1)
    
        return spikes.to(torch.uint8)
    
    def _reverse_no_last(self, x : torch.Tensor, max_y : torch.Tensor, mu : torch.Tensor, mult : float)->torch.Tensor:
        
        x = torch.round(self.get_gaussian(x, self.sigma2, mu) * mult) / mult
        mask = x < 0.1
        x[mask] = np.nan
        
        return x
    
    def _reverse(self, x : torch.Tensor, max_y : torch.Tensor, mu : torch.Tensor, mult : float)->torch.Tensor:
        
        x = torch.round(self.get_gaussian(x, self.sigma2, mu) * mult) / mult
        
        return x
    
    def _regular_no_last(self, x : torch.Tensor, max_y : torch.Tensor, mu : torch.Tensor, mult : float)->torch.Tensor:
        
        x = max_y - torch.round(self.get_gaussian(x, self.sigma2, mu) * mult) / mult 
        mask = x > max_y - 0.09
        x[mask] = torch.nan
        
        return x
    
    def _regular(self, x : torch.Tensor, max_y : torch.Tensor, mu : torch.Tensor, mult : float)->torch.Tensor:
        
        x = max_y - torch.round(self.get_gaussian(x, self.sigma2, mu) * mult) / mult
        
        return x
        

    def __call__(self, x : torch.Tensor, y : torch.Tensor)->tuple:
        
        
        assert x.min() >= 0, "Input values must be non-negative."
            
        h_mu = self.max_x / (self.n_fields - 1)

        max_y = torch.ceil(self.get_gaussian(h_mu, self.sigma2, h_mu))

        mu = torch.tile(torch.linspace(0, self.max_x, self.n_fields), (len(x[0]),)).to(self.device)
        
        x = torch.repeat_interleave(x, self.n_fields, dim=1).to(self.device)

        assert len(mu) == len(x[0])
        
        mult = 10 ** self.k_round
        
        x = self.transform_samples(x, max_y, mu, mult)

        x *= self.scale
            
        spikes = self.get_spikes(x,y,len(torch.unique(y)))
        
        return spikes, y
