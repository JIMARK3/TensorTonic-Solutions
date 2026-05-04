import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs) 
    L = max_len if max_len is not None else max(len(seq) for seq in seqs) 
    pad_array = np.full((N,L),fill_value=pad_value)
    for i,seq in enumerate(seqs):
        current_len = min(len(seq),L)
        pad_array[i,:current_len] = seq[:current_len]
    return pad_array
    