import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    total_loss = 0
    batch_count = 0
    
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()

        x , y = batch
        pred = model(x)

        loss = criterion(pred,y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1

    return total_loss/batch_count