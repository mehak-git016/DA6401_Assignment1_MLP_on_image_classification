"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative


class Layer:
    """
    A single fully-connected layer with activation function.
    forward: Applies linear transform + activation
    backward: Computes gradients
    It stores grad_W and grad_b after backward pass (it is required by autograder)
    """
    
    def __init__(self, input_size, output_size, activation='relu', weight_init='random'):
        """
        Initializing layer with weights and biases.
        Args used here:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: 'relu', 'sigmoid', 'tanh', or 'linear'
            weight_init: 'random' or 'xavier'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights based on strategy (random or Xavier)
        if weight_init == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else: 
            self.W = np.random.randn(input_size, output_size) * 0.01
        
        # Bias initialized to zero
        self.b = np.zeros((1, output_size))
        # Cache values for backward pass
        self.X = None
        self.Z = None  # pre-activation
        self.A = None  # post-activation 
        # Gradients (required by autograder)
        self.grad_W = None
        self.grad_b = None
    
    def _apply_activation(self, Z):
        """Applying activation function to pre-activation Z"""
        if self.activation == 'relu':
            return relu(Z)
        elif self.activation == 'sigmoid':
            return sigmoid(Z)
        elif self.activation == 'tanh':
            return tanh(Z)
        else:  # linear
            return Z
    
    def _activation_derivative(self, Z):
        """Get derivative of activation w.r.t. pre-activation Z"""
        if self.activation == 'relu':
            return relu_derivative(Z)
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(Z)
        elif self.activation == 'tanh':
            return tanh_derivative(Z)
        else:  # linear
            return np.ones_like(Z)
    
    def forward(self, X):
        """
        Forward pass: Z = XW + b, A = activation(Z)
        Args: X: Input of shape (batch_size, input_size)
        Returns: A: Activated output of shape (batch_size, output_size)
        """
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self._apply_activation(self.Z)
        return self.A
    
    def backward(self, dA):
        """
        Backward pass: gradients w.r.t. W, b, and input
        Args: dA: Gradient w.r.t. activation output of shape (batch_size, output_size)
        Returns: dX: Gradient w.r.t. input X of shape (batch_size, input_size)
        """
        # Gradient w.r.t. pre-activation Z
        dZ = dA * self._activation_derivative(self.Z)  # (batch, output_size)
        # Gradient w.r.t. weights
        self.grad_W = self.X.T @ dZ  # (input_size, output_size)
        # Gradient w.r.t. bias
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)  # (1, output_size)
        # Gradient w.r.t. input (for previous layer)
        dX = dZ @ self.W.T  # (batch, input_size)
        return dX
