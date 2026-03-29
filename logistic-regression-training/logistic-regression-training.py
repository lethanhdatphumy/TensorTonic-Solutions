import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    
    # Write code here
    for i in range(steps):
        y_pred = _sigmoid(X@w + b)

        loss = -1/len(X) * np.sum(y*np.log(y_pred)+ (1-y)*np.log(1-y_pred))

        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        w -= dw * lr
        b -= db * lr
    return (w,b)