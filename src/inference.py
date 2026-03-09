"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.objective_functions import cross_entropy_loss


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """

    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, default="best_model.npy",
                        help="Path to saved model weights (relative path)")
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

    args = parser.parse_args()
    args.hidden_layers = list(args.hidden_size)
    if args.num_layers != len(args.hidden_layers):
        args.num_layers = len(args.hidden_layers)
    return args


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
        
    Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    logits, _ = model.forward(X_test)

    loss, _ = cross_entropy_loss(y_test, logits)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """
    Main inference function.

    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    args = parse_arguments()

    print(f"Loading dataset {args.dataset}...")

    _, _, X_test, y_test = load_dataset(args.dataset)

    print("Loading model...")

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    best_weights = model.get_weights()
    np.save("best_model.npy", best_weights)

    print("Evaluating model...")

    metrics = evaluate_model(model, X_test, y_test)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("Evaluation complete!")

    return metrics


if __name__ == "__main__":
    main()
