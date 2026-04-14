import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = len(max(seqs, key=len))
    seqs = np.array(seqs, dtype=object)

    for i in range(len(seqs)):
        length = len(seqs[i])
        if length < max_len:
            add = [pad_value]*(max_len-length)
            seqs[i] = seqs[i] + add
        else:
            seqs[i] = seqs[i][:max_len]

    return np.array(seqs)