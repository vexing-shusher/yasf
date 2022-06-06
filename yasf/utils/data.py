import torch
from torch.utils.data import Dataset

import numpy as np
        
    
class DataWrapper(Dataset):
    
    def __init__(self, x, y, sample_transform=lambda x : x, target_transform=lambda x : x, global_transform=lambda x,y: (x,y) ):
        
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        #global_transform -- any function acting on the whole dataset
        self.ds = global_transform(x,y)
            
    def __len__(self):
        return len(self.ds[0])
    
    def __getitem__(self, idx):
        
        x = self.ds[0][idx]
        y = self.ds[1][idx]
        
        sample = self.sample_transform(x)
        target = self.target_transform(y)
        
        return (sample, target)
