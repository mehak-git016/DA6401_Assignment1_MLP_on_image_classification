"""
W&B Experiment 2.3: The Optimizer Showdown

Compare the convergence rates of all 4 optimizers using the same architecture
(3 hidden layers, 128 neurons each, ReLU activation). Which optimizer minimized
the loss fastest in the first 5 epochs? Theoretically, why does RMSProp often
outperform standard SGD on image classification?
"""

import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from ann.objective_functions import cross_entropy_loss
from utils.data_loader import load_dataset, create_batches
from sklearn.metrics import accuracy_score
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Optimizer Showdown Experiment')
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_assign1')
    return parser.parse_args()


def train_with_optimizer(optimizer_name, args):
    """Train model with specific optimizer"""
    epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # Initialize wandb run
    run_name = f"optimizer_{optimizer_name}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "optimizer": optimizer_name,
            "architecture": [128, 128, 128],
            "activation": "relu",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        }
    )

    # Load data
    X_train, y_train, X_test, y_test = load_dataset('mnist')

    # Create validation split
    X_val = X_train[-10000:]
    y_val = y_train[-10000:]
    X_train = X_train[:-10000]
    y_train = y_train[:-10000]

    # Create model (3 layers, 128 neurons each, ReLU)
    model_config = type('Config', (), {
        'num_layers': 3,
        'hidden_size': [128, 128, 128],
        'activation': 'relu',
        'weight_init': 'xavier',
        'loss': 'cross_entropy',
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'weight_decay': 0.0
    })()
    model = NeuralNetwork(model_config)

    # Get optimizer
    optimizer_config = type('Config', (), {
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'momentum': 0.9,
        'beta': 0.9,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    })()
    optimizer = get_optimizer(optimizer_config)

    loss_fn = cross_entropy_loss

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        # Training
        for X_batch, y_batch in create_batches(X_train, y_train,
                                             batch_size=batch_size, shuffle=True):
            # Forward pass
            logits, _ = model.forward(X_batch)

            # Compute loss and gradient
            loss, grad = loss_fn(y_batch, logits)

            # Backward pass
            model.backward(grad)

            # Prepare weights and gradients for optimizer
            weights = []
            gradients = []
            for layer in model.layers:
                weights.append(layer.W)
                weights.append(layer.b)
                gradients.append(layer.grad_W)
                gradients.append(layer.grad_b)

            # Update weights
            optimizer.update(weights, gradients)

            epoch_loss += loss
            num_batches += 1

        # Validation
        val_logits, _ = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = accuracy_score(val_true, val_preds)

        avg_loss = epoch_loss / num_batches

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_accuracy': val_acc,
            'optimizer': optimizer_name
        })

        print(f"{optimizer_name} - Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

    wandb.finish()


def main():
    args = parse_args()

    optimizers = ['sgd', 'momentum', 'nag', 'rmsprop']

    print("Starting Optimizer Showdown...")
    print("Comparing: SGD, Momentum, NAG, RMSProp")
    print("Architecture: 3 hidden layers × 128 neurons, ReLU activation")
    print("Convergence comparison window: first 5 epochs")

    for optimizer in optimizers:
        print(f"\nTraining with {optimizer.upper()}...")
        train_with_optimizer(optimizer, args)

    print("\nAll optimizer comparisons completed!")
    print("Check W&B dashboard for convergence comparison plots")


if __name__ == '__main__':
    main()
