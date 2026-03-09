"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

# ReLU (Rectified Linear Unit) and its derivative
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return (x>0).astype(float)

# Sigmoid and its derivative
def sigmoid(x):
    x = np.clip(x, -500, 500) # Clipping for numerical stability
    return (1/(1 + np.exp(-x)))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return (s*(1 - s))

# Tanh and its derivative
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    t = tanh(x)
    return (1-t**2)

# Softmax (for multi-class classification)
def softmax(logits):
    e_z = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    # Subtracting max for numerical stability (prevents overflow)
    return e_z / np.sum(e_z, axis=-1, keepdims=True)