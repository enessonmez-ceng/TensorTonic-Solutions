import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.p == 1.0:
            return torch.zeros_like(x)
            
        if self.training :
            
            m = (torch.rand_like(x) > self.p).to(x.dtype)
            return torch.mul(m,x)/(1-self.p)
        else:
            return x
        
