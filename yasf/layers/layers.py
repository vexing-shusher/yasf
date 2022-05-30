import torch
from torch.nn import Module
import numpy as np

class AbstractLayer(Module):
    
    def __init__(self,
                 n : int,
                 dt : float = 1.,
                 time : float = 300.,
                 store_all_spikes : bool = False,
                 device = torch.device("cpu"),
                )->None:
        
        self.device = device
    
        self.n = n
        self.dt = torch.as_tensor(dt, dtype=torch.float32, device = self.device)
        self.time = time

        self.store_all_spikes = store_all_spikes

        #initial spikes
        if self.store_all_spikes:
            self.s = torch.zeros((n, int(time/dt)), dtype=torch.uint8, device = self.device) #spike array
            self.t = torch.ones((n, int(time/dt)), dtype=torch.float32, device = self.device) * (-np.inf) #spike times array
        else:
            self.s = torch.zeros((n, 1), dtype=torch.uint8, device = self.device)
            self.t = torch.ones((n,1), dtype=torch.float32, device = self.device)
            
    def __call__(self,)->None:
        pass
    
    def reset_state(self):
        #initial spikes
        if self.store_all_spikes:
            self.s = torch.zeros((self.n, int(time/dt)), dtype=torch.uint8, device = self.device) #spike array
            self.t = torch.ones((self.n, int(time/dt)), dtype=torch.float32, device = self.device) * (-np.inf) #spike times array
        else:
            self.s = torch.zeros((self.n, 1), dtype=torch.uint8, device = self.device)
            self.t = torch.ones((self.n,1), dtype=torch.float32, device = self.device)                

class InputLayer(AbstractLayer):
    
    def __init__(
        self,
        n: int,
        dt : float = 1.,
        time : float = 300,
        store_all_spikes : bool = False,
        device = torch.device("cpu"),
    )->None:
        
        super().__init__(n = n,
                         dt = dt,
                         time = time,
                         store_all_spikes = store_all_spikes,
                         device = device)

        
    def __call__(self, inp : torch.Tensor, ts : int = 0)->tuple:
        
        '''
        inp -- input spike train
        w -- connection weights' matrix
        ts -- time step (if we store all spikes, this argument is required!)
        '''
        
        assert inp.device == self.device, f"Input must be on {self.device}."
        
        #record spikes
        self.s[:,ts] = inp
        
        #record spiking times
        self.t[:,ts].masked_fill_(self.s[:,ts].to(torch.bool), ts * self.dt / self.time) 
        
        
        return self.s[:,ts], self.t[:,ts]

class LIFLayer(AbstractLayer):
    
    def __init__(
        self,
        n: int,
        dt : float = 1.,
        time : float = 300,
        rest : float = 0.,
        reset : float = 0.,
        thresh : float = 10,
        refrac : float = 1.,
        tc_decay : float = 100.,
        store_all_spikes : bool = False,
        device = torch.device("cpu"),
    )->None:
        
        super().__init__(n = n,
                         dt = dt,
                         time = time,
                         store_all_spikes = store_all_spikes,
                         device = device)
    
        
        self.rest = torch.as_tensor(rest,dtype=torch.float32, device=self.device)
        self.reset = torch.as_tensor(reset,dtype=torch.float32, device=self.device)
        self.thresh = torch.as_tensor(thresh,dtype=torch.float32, device=self.device)
        self.refrac = torch.as_tensor(refrac,dtype=torch.float32, device=self.device)
        
        self.decay = torch.exp(torch.as_tensor(-dt/tc_decay,dtype=torch.float32, device=self.device))
        
        #initial voltages
        self.v = torch.ones(n, device = self.device, dtype=torch.float32) * self.reset

        #refractory states of the neurons in the layer
        self.rt = torch.zeros(n, device = self.device, dtype=torch.float32)
        
    def __call__(self, inp : torch.Tensor, 
                 w: torch.Tensor, 
                 w_rec : torch.Tensor,
                 v_ext = 0,
                 ts : int = -1)->tuple:
        
        '''
        inp -- input spike train
        w -- connection weights' matrix
        w_rec -- interlayer connections. required to avoid using "if-else" constructions
        v_ext -- external voltage, can be used to transmit teacher signal to the layer
        ts -- time step (if we store all spikes, this argument is required!)
        '''
        
        
        assert inp.device == self.device, f"Input must be on {self.device}."
        assert w.device == self.device, f"Weights must be on {self.device}."
        assert w_rec.device == self.device, f"Recurrent weights must be on {self.device}."
        
        #v_ext = torch.mul(v_ext,torch.ones(self.n, dtype=torch.float32, device=self.device))
        
        #print(v_ext)

        #calculate activations: multiply input spike trains by the connection weight matrix
        x = inp.view(1,w.shape[0]).to(torch.float32) @ w + self.s[:,-1].view(1,w_rec.shape[0]).to(torch.float32) @ w_rec
        x = x.squeeze()
        
        #add teacher voltage
        self.v += v_ext
        
        #decay voltages
        self.v = self.decay * (self.v - self.rest) + self.rest
        
        #integrate inputs: if refractory time of the neuron is > 0, ignore input
        x.masked_fill_(self.rt > 0, 0.)
        
        #decrement refractory times
        self.rt -= self.dt
        
        #increase voltages
        self.v += x
        
        #check for spiking neurons
        self.s[:,ts] = (self.v >= self.thresh).to(self.device)
        
        #record spiking times
        self.t[:,ts].masked_fill_(self.s[:,ts].to(torch.bool), ts * self.dt / self.time)
        
        #reset refractory times
        self.rt.masked_fill_(self.s[:,ts].to(torch.bool),self.refrac)
        
        #reset voltages
        self.v.masked_fill_(self.s[:,ts].to(torch.bool),self.reset)
        
        return self.s[:,ts], self.t[:,ts]
    
    def reset_state(self)->None:
        #initial spikes
        if self.store_all_spikes:
            self.s = torch.zeros((self.n, int(time/dt)), dtype=torch.uint8, device = self.device) #spike array
            self.t = torch.ones((self.n, int(time/dt)), dtype=torch.float32, device = self.device) * (-np.inf) #spike times array
        else:
            self.s = torch.zeros((self.n, 1), dtype=torch.uint8, device = self.device)
            self.t = torch.ones((self.n,1), dtype=torch.float32, device = self.device)      

        #initial voltages
        self.v = torch.ones(self.n, device = self.device, dtype=torch.float32) * self.reset

        #refractory states of the neurons in the layer
        self.rt = torch.zeros(self.n, device = self.device, dtype=torch.float32)

class DaCLayer(AbstractLayer):
    
    def __init__(
        self,
        n: int,
        dt : float = 1.,
        time : float = 300,
        rest : float = 0.,
        reset : float = 0.,
        thresh : float = 10,
        refrac : float = 1.,
        tc_decay : float = 100.,
        theta_plus : float = 0.05,
        tc_theta_decay : float = 1e7,
        store_all_spikes : bool = False,
        device = torch.device("cpu"),
    )->None:
        
        super().__init__(n = n,
                         dt = dt,
                         time = time,
                         store_all_spikes = store_all_spikes,
                         device = device)
    
        
        self.rest = torch.as_tensor(rest,dtype=torch.float32, device=self.device)
        self.reset = torch.as_tensor(reset,dtype=torch.float32, device=self.device)
        self.thresh = torch.as_tensor(thresh,dtype=torch.float32, device=self.device)
        self.refrac = torch.as_tensor(refrac,dtype=torch.float32, device=self.device)
        self.theta_plus = torch.as_tensor(theta_plus,dtype=torch.float32, device=self.device)
        
        self.decay = torch.exp(torch.as_tensor(-dt/tc_decay,dtype=torch.float32, device=self.device))
        self.theta_decay = torch.exp(torch.as_tensor(-dt/tc_theta_decay,dtype=torch.float32, device=self.device))
        
        #initial voltages
        self.v = torch.ones(n, device = self.device, dtype=torch.float32) * self.reset

        #refractory states of the neurons in the layer
        self.rt = torch.zeros(n, device = self.device, dtype=torch.float32)
        
        #adaptive thresholds
        self.theta = torch.zeros(n, device = self.device, dtype=torch.float32)
        


    def __call__(self, inp : torch.Tensor, 
                 w: torch.Tensor, 
                 w_rec : torch.Tensor,
                 v_ext = 0,
                 ts : int = -1)->tuple:
        
        assert inp.device == self.device, f"Input must be on {self.device}."
        assert w.device == self.device, f"Weights must be on {self.device}."
        assert w_rec.device == self.device, f"Recurrent weights must be on {self.device}."
        

        #calculate activations: multiply input spike trains by the connection weight matrix
        x = inp.view(1,w.shape[0]).to(torch.float32) @ w + self.s[:,-1].view(1,w_rec.shape[0]).to(torch.float32) @ w_rec
        x = x.squeeze()
        
        #add teacher voltage
        self.v += v_ext
        
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        
        self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.rt <= 0).float() * x

        # Decrement refractory counters.
        self.rt -= self.dt

        # Check for spiking neurons.
        self.s[:,ts] = (self.v >= self.thresh + self.theta).to(self.device)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.rt.masked_fill_(self.s[:,ts].to(torch.bool), self.refrac)
        self.v.masked_fill_(self.s[:,ts].to(torch.bool), self.reset)
        
        self.theta += self.theta_plus * self.s[:,ts].float().sum(0)
        
        return self.s[:,ts], self.t[:,ts]
    
    def reset_state(self)->None:
        #initial spikes
        if self.store_all_spikes:
            self.s = torch.zeros((self.n, int(time/dt)), dtype=torch.uint8, device = self.device) #spike array
            self.t = torch.ones((self.n, int(time/dt)), dtype=torch.float32, device = self.device) * (-np.inf) #spike times array
        else:
            self.s = torch.zeros((self.n, 1), dtype=torch.uint8, device = self.device)
            self.t = torch.ones((self.n,1), dtype=torch.float32, device = self.device)      

        #initial voltages
        self.v = torch.ones(self.n, device = self.device, dtype=torch.float32) * self.reset

        #refractory states of the neurons in the layer
        self.rt = torch.zeros(self.n, device = self.device, dtype=torch.float32)
        
        #adaptive thresholds
        self.theta = torch.zeros(self.n, device = self.device, dtype=torch.float32)
