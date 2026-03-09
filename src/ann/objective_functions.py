"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def cross_entropy_loss(y_true, logits):
    from .activations import softmax
    prob = softmax(logits)
    m = y_true.shape[0]
    epsilon = 1e-12
    loss = -np.sum(y_true*np.log(prob+epsilon)) / m #y_true is one-hot encoded
    grad = (prob-y_true) / m #gradient of loss wrt logits
    return loss, grad

def mse_loss(y_true, y_pred):
    n = y_true.size  # mean over all elements
    loss = np.mean((y_true - y_pred)**2)
    grad = (2/n) * (y_pred - y_true)
    return loss, grad
