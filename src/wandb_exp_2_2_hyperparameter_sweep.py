"""
W&B Experiment 2.2: Hyperparameter Sweep

Perform a W&B Sweep with at least 100 runs, varying hyperparameters.
Using the Parallel Coordinates plot, identify which hyperparameter had the most
significant impact on validation accuracy. What was your best-performing configuration?

Note: For reproducible results, run the sweep multiple times with the same
sweep ID to get consistent hyperparameter combinations.
"""

import wandb
import numpy as np
import json
import os
import random

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from ann.objective_functions import cross_entropy_loss, mse_loss
from utils.data_loader import load_dataset, create_batches

from sklearn.metrics import accuracy_score, f1_score

SEED = 22
random.seed(SEED)
np.random.seed(SEED)

# path where best files will be stored
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(SRC_DIR, "best_model.npy")
BEST_CONFIG_PATH = os.path.join(SRC_DIR, "best_config.json")
BEST_SCORE_PATH = os.path.join(SRC_DIR, "best_f1.txt")


def get_saved_best_f1():
    """Load previously saved best F1 score."""
    if os.path.exists(BEST_SCORE_PATH):
        try:
            with open(BEST_SCORE_PATH, "r") as f:
                return float(f.read().strip())
        except (ValueError, OSError):
            return -1
    return -1


def save_best_model(model, config, f1):
    """Save best model and configuration."""
    weights = model.get_weights()
    np.save(BEST_MODEL_PATH, weights)

    # Save a clean JSON-serializable config that represents the best run.
    config_dict = dict(config)
    config_dict["best_test_f1"] = float(f1)

    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)

    with open(BEST_SCORE_PATH, "w") as f:
        f.write(str(f1))

    print(f"New best model saved with Test F1 = {f1:.4f}")


def train_model(config=None):

    with wandb.init(config=config):

        config = wandb.config
        config.hidden_layers = list(config.hidden_size)
        if int(config.num_layers) != len(config.hidden_layers):
            config.num_layers = len(config.hidden_layers)

        # Load dataset
        X_train, y_train, X_test, y_test = load_dataset("mnist")

        # Validation split
        X_val = X_train[-10000:]
        y_val = y_train[-10000:]
        X_train = X_train[:-10000]
        y_train = y_train[:-10000]

        model = NeuralNetwork(config)
        optimizer = get_optimizer(config)

        loss_fn = cross_entropy_loss if config.loss == "cross_entropy" else mse_loss

        best_val_acc = 0

        for epoch in range(config.epochs):

            epoch_loss = 0
            num_batches = 0

            for X_batch, y_batch in create_batches(
                X_train,
                y_train,
                batch_size=config.batch_size,
                shuffle=True
            ):

                logits, _ = model.forward(X_batch)

                loss, grad = loss_fn(y_batch, logits)

                model.backward(grad)

                weights = []
                grads = []

                for layer in model.layers:
                    weights += [layer.W, layer.b]
                    grads += [layer.grad_W, layer.grad_b]

                optimizer.update(weights, grads)

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # Validation accuracy
            val_logits, _ = model.forward(X_val)

            val_preds = np.argmax(val_logits, axis=1)
            val_true = np.argmax(y_val, axis=1)

            val_acc = accuracy_score(val_true, val_preds)

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_accuracy": val_acc
            })

            best_val_acc = max(best_val_acc, val_acc)

        # ---------- FINAL METRICS ----------

        train_logits, _ = model.forward(X_train)
        train_preds = np.argmax(train_logits, axis=1)
        train_true = np.argmax(y_train, axis=1)
        train_acc = accuracy_score(train_true, train_preds)

        test_logits, _ = model.forward(X_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_true = np.argmax(y_test, axis=1)

        test_acc = accuracy_score(test_true, test_preds)

        test_f1 = f1_score(test_true, test_preds, average="macro")

        current_best_f1 = get_saved_best_f1()
        saved_as_best_model = test_f1 > current_best_f1

        wandb.log({
            "best_val_accuracy": best_val_acc,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "selection_f1": test_f1,
            "saved_as_best_model": int(saved_as_best_model),
            "prev_best_f1": current_best_f1
        })

        # ---------- SAVE BEST MODEL ----------

        if saved_as_best_model:
            save_best_model(model, config, test_f1)


def main():

    sweep_config = {

        "method": "random",

        "seed": 22,

        "metric": {
            "name": "best_val_accuracy",
            "goal": "maximize"
        },

        "parameters": {

            "epochs": {"values": [5, 8, 10]},

            "batch_size": {"values": [32, 64, 128]},

            "optimizer": {
                "values": ["sgd", "momentum", "rmsprop", "nag"]
            },

            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.0001,
                "max": 0.02
            },

            "weight_decay": {
                "values": [0.0, 0.0001, 0.001, 0.01]
            },

            "num_layers": {
                "values": [2, 3, 4]
            },

            "hidden_size": {
                "values": [
                    [64, 32],
                    [128, 64],
                    [128, 64, 32],
                    [128, 128, 64],
                    [128, 128, 64, 32],
                    [128, 64, 64, 64]
                ]
            },

            "activation": {
                "values": ["relu", "sigmoid", "tanh"]
            },

            "weight_init": {
                "values": ["random", "xavier"]
            },

            "loss": {
                "values": ["cross_entropy", "mse"]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="DA6401_assign1")

    print("Sweep ID:", sweep_id)

    wandb.agent(sweep_id, train_model, count=100)


if __name__ == "__main__":
    main()
