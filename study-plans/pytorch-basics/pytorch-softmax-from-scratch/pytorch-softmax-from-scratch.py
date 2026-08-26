import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    logit_t = torch.tensor(logits)
    m = torch.max(logit_t,dim=1, keepdim=True).values
    shifted = logit_t - m
    exps = torch.exp(shifted)

    return exps / exps.sum(dim = 1 , keepdim = True)
