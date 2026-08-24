import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x_t = torch.tensor(x,dtype = torch.float32)
    
    if op == "flatten":
        return torch.flatten(x_t).tolist()
    elif op == "squeeze":
        return torch.squeeze(x_t).tolist()
    elif op == "transpose":
        return x_t.T.tolist()
    else:
        raise ValueError("Invalid operation")