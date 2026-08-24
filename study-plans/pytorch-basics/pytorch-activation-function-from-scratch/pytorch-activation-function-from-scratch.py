import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    
    x_t = torch.tensor(x,dtype = torch.float32)
    
    methods = {
        "relu" : lambda t : torch.where(t>0, t , 0.0),
        "sigmoid" : lambda t : 1/(1+ torch.exp(-t)),
        "tanh" : lambda t : (torch.exp(t) - torch.exp(-t)) / (torch.exp(t) + torch.exp(-t)),
        "leaky_relu" : lambda t : torch.where(t>0 , t , 0.01*t)
    }

    if method not in methods:
        raise ValueError("Invalid method")

    return methods[method](x_t).tolist()