"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, create_batches
from ann.objective_functions import cross_entropy_loss, mse_loss

from sklearn.metrics import accuracy_score, f1_score


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="mse",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", "--learning-rate", type=float, default=0.003110199715274576)
    parser.add_argument("-wd", "--weight_decay", "--weight-decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", "--num-layers", type=int, default=4)
    parser.add_argument("-sz", "--hidden_size", "--hidden-size", type=int, nargs="+",
                        default=[128, 64, 64, 64])
    parser.add_argument("-a", "--activation", type=str, default="tanh",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-wi", "--weight_init", "--weight-init", type=str, default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-wp", "--wandb_project", "--wandb-project", type=str,
                        default="DA6401_assign1")
    parser.add_argument("--model_save_path", type=str, default="best_model.npy")
    parser.add_argument("--model_path", type=str, default="best_model.npy")

    args = parser.parse_args()
    args.hidden_layers = list(args.hidden_size)
    if args.num_layers != len(args.hidden_layers):
        args.num_layers = len(args.hidden_layers)
    return args


def main():
    """
    Main training function.
    """

    args = parse_arguments()

    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    print("Building model...")

    model = NeuralNetwork(args)

    if args.loss == "cross_entropy":
        loss_fn = cross_entropy_loss
    else:
        loss_fn = mse_loss

    best_f1 = -1
    best_weights = None

    print("Starting training...\n")

    for epoch in range(args.epochs):

        epoch_loss = 0
        num_batches = 0

        for X_batch, y_batch in create_batches(
                X_train, y_train, batch_size=args.batch_size, shuffle=True):

            logits, _ = model.forward(X_batch)

            loss, grad = loss_fn(y_batch, logits)

            model.backward(grad)

            model.update_weights()

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # evaluation
        test_logits, _ = model.forward(X_test)

        preds = np.argmax(test_logits, axis=1)
        labels = np.argmax(y_test, axis=1)

        test_acc = accuracy_score(labels, preds)
        test_f1 = f1_score(labels, preds, average="macro")

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Loss: {avg_loss:.6f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}\n")

        # save best model
        if test_f1 > best_f1:

            best_f1 = test_f1

            best_weights = model.get_weights()

            np.save(args.model_save_path, best_weights)

            config_path = "best_config.json"

            with open(config_path, "w") as f:
                json.dump(vars(args), f, indent=4)

            print(f"Saved best model with F1: {best_f1:.4f}")

    print("\nTraining complete!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Model saved at: {args.model_save_path}")

if __name__ == "__main__":
    main()
