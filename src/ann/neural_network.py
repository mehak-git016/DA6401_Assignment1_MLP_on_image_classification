"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
import os
import sys
from pathlib import Path
from .neural_layer import Layer
from .objective_functions import cross_entropy_loss, mse_loss
from .optimizers import SGD, Momentum, NAG, RMSProp


class ForwardOutput:
    """
    Array-like wrapper for logits that is also unpackable as (logits, cache).
    This keeps compatibility with both:
      1) logits = model.forward(X)           # autograder style
      2) logits, _ = model.forward(X)        # existing project style
    """

    def __init__(self, logits, cache=None):
        self.logits = logits
        self.cache = cache

    def __iter__(self):
        yield self.logits
        yield self.cache

    def __array__(self, dtype=None):
        return np.asarray(self.logits, dtype=dtype)

    def __getattr__(self, name):
        return getattr(self.logits, name)

    def __getitem__(self, key):
        return self.logits[key]

    def __repr__(self):
        return repr(self.logits)


def _ensure_test_workdir():
    """
    Keep src/test.py compatible when invoked from project root.
    """
    if os.path.basename(sys.argv[0]) != "test.py":
        return

    if (Path.cwd() / "best_model.npy").exists():
        return

    src_dir = Path(__file__).resolve().parents[1]
    if (src_dir / "best_model.npy").exists():
        os.chdir(src_dir)


_ensure_test_workdir()


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):

        self.cli_args = cli_args
        self.layers = []
        self._printed_grad_shapes = False

        input_dim = 784
        output_dim = 10

        hidden_sizes = getattr(cli_args, "hidden_layers", None)
        if hidden_sizes is None:
            hidden_sizes = getattr(cli_args, "hidden_size", None)
        if hidden_sizes is None:
            raise ValueError("Missing hidden layer sizes: expected 'hidden_layers' or 'hidden_size'")
        activation = cli_args.activation
        weight_init = cli_args.weight_init

        layer_dims = [input_dim] + hidden_sizes + [output_dim]

        for i in range(len(layer_dims) - 1):

            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            if i < len(layer_dims) - 2:
                act = activation
            else:
                act = "linear"

            layer = Layer(
                input_size=in_dim,
                output_size=out_dim,
                activation=act,
                weight_init=weight_init
            )

            self.layers.append(layer)

        # loss
        if cli_args.loss == "cross_entropy":
            self.loss_fn = cross_entropy_loss
        else:
            self.loss_fn = mse_loss

        # optimizer
        lr = getattr(cli_args, "learning_rate", None)
        if lr is None:
            lr = getattr(cli_args, "lr", None)
        if lr is None:
            raise ValueError("Missing learning rate: expected 'learning_rate' or 'lr'")

        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(lr)

        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(lr)

        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(lr)

        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(lr)
        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns:
            logits: raw output scores (no softmax applied)
            cache: placeholder for compatibility
        """

        A = X

        for layer in self.layers:
            A = layer.forward(A)

        return ForwardOutput(A, None)

    def backward(self, y_true_or_grad, y_pred=None):
        """
        Backward propagation to compute gradients.

        grad_Ws[0] = last layer
        grad_bs[0] = last layer
        """

        if y_pred is None:
            grad = y_true_or_grad
        else:
            y_true = y_true_or_grad
            y_pred = np.asarray(y_pred)
            # Support both class-index labels and one-hot labels.
            if (
                y_pred.ndim == 2
                and (
                    y_true.ndim == 1
                    or (y_true.ndim == 2 and y_true.shape[1] == 1)
                )
            ):
                y_idx = np.asarray(y_true).reshape(-1).astype(int)
                y_one_hot = np.zeros_like(y_pred)
                y_one_hot[np.arange(y_pred.shape[0]), y_idx] = 1.0
                y_true = y_one_hot
            _, grad = self.loss_fn(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        dA = grad

        for layer in reversed(self.layers):

            dA = layer.backward(dA)

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        if not self._printed_grad_shapes:
            if self.grad_W.size > 1 and self.grad_b.size > 1:
                print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
                print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
            elif self.grad_W.size > 0 and self.grad_b.size > 0:
                print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[0].shape)
                print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[0].shape)
            else:
                print("Shape of grad_Ws:", self.grad_W.shape)
                print("Shape of grad_bs:", self.grad_b.shape)
            self._printed_grad_shapes = True

        return self.grad_W, self.grad_b

    def update_weights(self):
        """
        Update weights using optimizer
        """

        weights = []
        gradients = []
        weight_decay = float(getattr(self.cli_args, "weight_decay", 0.0))

        for layer, gw, gb in zip(reversed(self.layers), self.grad_W, self.grad_b):

            weights.append(layer.W)
            weights.append(layer.b)

            if weight_decay > 0:
                gradients.append(gw + weight_decay * layer.W)
            else:
                gradients.append(gw)
            gradients.append(gb)

        self.optimizer.update(weights, gradients)

    def train(self, X_train, y_train, epochs=1, batch_size=32):

        n = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n)
            X_train = X_train[indices]
            y_train = y_train[indices]

            total_loss = 0

            for start in range(0, n, batch_size):

                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                logits, _ = self.forward(X_batch)

                loss, _ = self.loss_fn(y_batch, logits)

                self.backward(y_batch, logits)

                self.update_weights()

                total_loss += loss

            avg_loss = total_loss / (n // batch_size + 1)

            print("Epoch:", epoch + 1, "Loss:", avg_loss)

    def evaluate(self, X, y):

        logits, _ = self.forward(X)

        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)

        accuracy = np.mean(preds == labels)

        return accuracy

    def get_weights(self):

        d = {}

        for i, layer in enumerate(self.layers):

            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        return d

    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
