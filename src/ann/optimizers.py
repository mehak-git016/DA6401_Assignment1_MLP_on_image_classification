"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np

class SGD:
    """Simple Gradient Descent
    uses: weight = weight - learning_rate * gradient"""
    def __init__(self, learning_rate=0.02):
        self.lr = learning_rate
    def lookahead(self, weights):
        """For SGD, lookahead is just current weights . 
        This is to be consistent with NAG"""
        return weights
    def update(self, weights, gradients):
        for w,g in zip(weights, gradients):
            w -= self.lr*g
        return weights
    
class Momentum:
    """SGD with Momentum (speeds up in consistent favourable direction)
    uses: velocity = momentum*velocity + learning_rate*gradient
          weight = weight - velocity"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = None
    def lookahead(self, weights):
        return weights
    def update(self, weights, gradients):
        # Initializing velocities
        if self.velocities is None:
            self.velocities = [np.zeros_like(w) for w in weights]
        for i,(w,g) in enumerate(zip(weights,gradients)):
            # Building momentum by accumulating past gradients
            self.velocities[i] = self.momentum*self.velocities[i] + self.lr*g
            # adding velocities to weight
            w -= self.velocities[i]
        return weights

class NAG:
    """
    Nesterov Accelerated Gradient optimizer.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = None
    def lookahead(self, weights):
        """
        Returns lookahead weights used for forward/backward pass
        """
        if self.velocities is None:
            self.velocities = [np.zeros_like(w) for w in weights]
        lookahead_weights = []
        for w, v in zip(weights, self.velocities):
            w_look = w - self.momentum * v
            lookahead_weights.append(w_look)
        return lookahead_weights
    def update(self, weights, gradients):
        """
        Update weights using gradients computed at lookahead position
        """
        if self.velocities is None:
            self.velocities = [np.zeros_like(w) for w in weights]
        for i, (w, g) in enumerate(zip(weights, gradients)):
            # update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * g
            # update weight
            weights[i] -= self.velocities[i]
        return weights

class RMSProp:
    """Root Mean Square Propagation: adaptive learning rates for each parameter
    uses: mean_sq_grad = decay_rate*mean_sq_grad + (1-decay_rate)*gradient^2
          weight = weight - (learning_rate / sqrt(mean_sq_grad + epsilon)) * gradient"""
    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.mean_sq_grads = None
    def lookahead(self, weights):
        return weights
    def update(self, weights, gradients):
        # Initializing mean squared gradients
        if self.mean_sq_grads is None:
            self.mean_sq_grads = [np.zeros_like(w) for w in weights]
        for i,(w,g) in enumerate(zip(weights,gradients)):
            # Updating mean squared gradient with decay
            self.mean_sq_grads[i] = self.decay*self.mean_sq_grads[i] + (1-self.decay)*g**2
            adaptive_lr = self.lr / (np.sqrt(self.mean_sq_grads[i]) + self.epsilon)
            w -= adaptive_lr * g
        return weights


def get_optimizer(config):
    """Factory function to create optimizer based on config"""
    optimizer_name = config.optimizer.lower()

    if optimizer_name == 'sgd':
        return SGD(learning_rate=config.learning_rate)
    elif optimizer_name == 'momentum':
        return Momentum(learning_rate=config.learning_rate,
                       momentum=getattr(config, 'momentum', 0.9))
    elif optimizer_name == 'nag':
        return NAG(learning_rate=config.learning_rate,
                  momentum=getattr(config, 'momentum', 0.9))
    elif optimizer_name == 'rmsprop':
        return RMSProp(learning_rate=config.learning_rate,
                      decay=getattr(config, 'beta', 0.9),
                      epsilon=getattr(config, 'epsilon', 1e-8))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")