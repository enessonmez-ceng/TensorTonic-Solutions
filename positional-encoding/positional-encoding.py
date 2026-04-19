import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(0,seq_len,1)[:, np.newaxis]
    i = np.arange(0,d_model,1)[np.newaxis , :]

    angles = pos / (base ** (2 * (i//2) / d_model))

    pe = np.zeros((seq_len,d_model))
    pe[:, 0::2] = np.sin(angles[: , 0::2])  
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe