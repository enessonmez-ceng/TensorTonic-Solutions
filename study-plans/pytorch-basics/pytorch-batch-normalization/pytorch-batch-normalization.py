import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    
    mean = X.mean(dim=0)
    var = X.var(dim=0, unbiased = False)

    X_t = torch.tensor(X)

    X_y = (X_t-mean)/torch.sqrt(var+eps)
    
    Y = gamma*X_y + beta
    return Y
