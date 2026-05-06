import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    a = np.array(A)
    ROW, COL = a.shape
    t = np.zeros((COL,ROW),dtype=a.dtype)
    for i in range(ROW):
        t[:,i] = a[i,:]
    return t
