import torch
from torch.utils.data import Dataset

import numpy as np
        
    
class DataWrapper(Dataset):
    
    def __init__(self, x, y, sample_transform = None, target_transform=None, global_transform=None):
        
      if sample_transform:  
        self.sample_transform = sample_transform
      else:
        self.sample_transform = lambda x : x

      if sample_transform:  
        self.target_transform = target_transform
      else:
        self.target_transform = lambda x : x

      #global_transfrom -- any function acting on the whole dataset
      if global_transform:
        self.ds = self.global_transform(x,y)
      else:
        self.ds = (x,y)
            
    def __len__(self):
        return len(self.ds[0])
    
    def __getitem__(self, idx):
        
        x = self.ds[0][idx]
        y = self.ds[1][idx]
        
        sample = self.sample_transform(x)
        target = self.target_transform(y)
        
        return (sample, target)
