import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype = torch.float32 , requires_grad = True)
    
    
    for i in range(0,len(micro_batches),accum_steps):
        window = micro_batches[i:i+accum_steps]
    
        for x,t in window:
            x_t = torch.tensor(x, dtype = torch.float32)
            t_t = torch.tensor(t, dtype = torch.float32)
            loss = (w@x_t-t_t)**2
    
            loss.backward()
    
        with torch.no_grad():
            
            grad_mean = w.grad/accum_steps
            w -= lr*grad_mean
            
        w.grad.zero_()

    return (w.tolist(),grad_mean.tolist())
    