import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    # Write code here
    x = np.array(X)                 # x (N, D)
    y = np.array(y)                 # y (N,  )
    N, D = x.shape
    
    w = np.zeros(D)                 # w (D， )
    b = 0.0            
    """
    p = sigmoid(xw+b)
    """
    for _ in range(steps):
        p = _sigmoid(np.dot(x, w) + b)      # p  (N, 1)
        dw = x.T @ (p - y) / N              # dw (D, 1)
        db = (p - y).mean()                 # db (1,  )
        w = w - lr * dw
        b = b - lr * db
    return (w, b.item())
    