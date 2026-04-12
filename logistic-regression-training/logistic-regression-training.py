import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, d = X.shape
    
    w = np.zeros(d)
    b = 0

    for _ in range(steps):
        # forward pass
        z = np.dot(X, w) + b
        predictions = _sigmoid(z)
        
        # gradients
        error = predictions - y

        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)

        # update
        w -= lr * dw
        b -= lr * db

    return w, b