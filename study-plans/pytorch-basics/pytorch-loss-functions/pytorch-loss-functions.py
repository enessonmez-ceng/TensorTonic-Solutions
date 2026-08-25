import torch
import torch.nn.functional as F

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    if method == "mse":
        pred_t = torch.tensor(pred).float()
        target_t = torch.tensor(target).float()
        return ((pred_t - target_t) ** 2).mean().item()
        
    elif method == "cross_entropy":
        
        logits = torch.tensor(pred).float()
        
        target_t = torch.tensor(target, dtype=torch.long)
        
        log_probs = F.log_softmax(logits, dim=1)
        
        
        batch_indices = torch.arange(logits.size(0))
        selected_log_probs = log_probs[batch_indices, target_t]
        
        return -selected_log_probs.mean().item()
        
    elif method == "huber":
        
        pred_t = torch.tensor(pred).float()
        target_t = torch.tensor(target).float()
        
        a = (pred_t - target_t).abs()
        delta_t = torch.tensor(delta).float()
        
        loss_tensor = torch.where(a > delta_t, delta_t * (a - 0.5 * delta_t), 0.5 * (a ** 2))
        return loss_tensor.mean().item()
        
    else:
        raise ValueError("Invalid method")
