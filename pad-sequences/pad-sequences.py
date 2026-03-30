import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    seqs = [list(s) for s in seqs]
    if max_len is None:
        max_len = len(max(seqs, key=len)) if seqs else 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        else:
            miss = max_len - len(seqs[i])
            paddings = np.full(miss, fill_value=pad_value)
            seqs[i] = np.append(seqs[i], paddings)
    return np.array(seqs)