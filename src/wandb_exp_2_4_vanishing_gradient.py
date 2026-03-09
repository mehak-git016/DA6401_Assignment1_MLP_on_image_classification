"""
W&B Experiment 2.4: Vanishing Gradient Analysis

Fix the optimizer to RMSProp and compare Sigmoid and ReLU for different network
configurations. Log the gradient norms for the first hidden layer. Do you observe the
vanishing gradient problem with Sigmoid? Provide a plot to support your observation.
"""

import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from ann.objective_functions import cross_entropy_loss
from utils.data_loader import load_dataset, create_batches
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Vanishing Gradient Analysis')
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_assign1')
    return parser.parse_args()


def train_with_activation(activation_name, network_depth, args):
    """Train model with specific activation and depth"""

    run_name = f"vanishing_grad_{activation_name}_depth_{network_depth}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "activation": activation_name,
            "network_depth": network_depth,
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 10
        }
    )

    config = wandb.config

    # Load data
    X_train, y_train, X_test, y_test = load_dataset('mnist')

    # Create validation split
    X_val = X_train[-5000:]
    y_val = y_train[-5000:]
    X_train = X_train[:-5000]
    y_train = y_train[:-5000]

    # Create model with varying depth
    hidden_sizes = [64] * network_depth  # Same width, varying depth
    model_config = type('Config', (), {
        'num_layers': network_depth,
        'hidden_size': hidden_sizes,
        'activation': activation_name,
        'weight_init': 'xavier',
        'loss': 'cross_entropy',
        'optimizer': 'rmsprop',
        'learning_rate': 0.001,
        'weight_decay': 0.0
    })()
    model = NeuralNetwork(model_config)

    # RMSProp optimizer
    optimizer_config = type('Config', (), {
        'optimizer': 'rmsprop',
        'learning_rate': 0.001,
        'beta': 0.9,
        'epsilon': 1e-8
    })()
    optimizer = get_optimizer(optimizer_config)

    loss_fn = cross_entropy_loss

    # Training loop
    for epoch in range(10):
        epoch_loss = 0
        num_batches = 0
        grad_norms_first_layer = []

        # Training
        for X_batch, y_batch in create_batches(X_train, y_train,
                                             batch_size=64, shuffle=True):
            # Forward pass
            logits, _ = model.forward(X_batch)

            # Compute loss and gradient
            loss, grad = loss_fn(y_batch, logits)

            # Backward pass
            model.backward(grad)

            # Track gradient norm for first hidden layer (layers are stored input->output)
            first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)
            grad_norms_first_layer.append(first_layer_grad_norm)

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

        # Calculate average gradient norm for this epoch
        avg_grad_norm = np.mean(grad_norms_first_layer)
        avg_loss = epoch_loss / num_batches

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'first_layer_grad_norm': avg_grad_norm,
            'activation': activation_name,
            'network_depth': network_depth
        })

        print(f"{activation_name} (depth {network_depth}) - Epoch {epoch}: "
              f"Loss={avg_loss:.4f}, Grad Norm={avg_grad_norm:.6f}")

    wandb.finish()


def main():
    args = parse_args()

    activations = ['sigmoid', 'relu']
    depths = [2, 4, 6]  # Different network depths

    print("Starting Vanishing Gradient Analysis...")
    print("Comparing Sigmoid vs ReLU across different network depths")
    print("Optimizer: RMSProp, Tracking first layer gradient norms")

    for activation in activations:
        for depth in depths:
            print(f"\n🏃 Training {activation.upper()} with depth {depth}...")
            train_with_activation(activation, depth, args)

    print("\nVanishing gradient analysis completed!")
    print("Check W&B dashboard for gradient norm plots")
    print("Look for: Sigmoid gradients vanishing with deeper networks")


if __name__ == '__main__':
    main()
