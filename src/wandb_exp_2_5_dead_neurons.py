"""
W&B Experiment 2.5: The "Dead Neuron" Investigation

Using ReLU activation and a high learning rate (e.g., 0.1), monitor the activations of
your hidden layers. Find a run where the validation accuracy plateaus early. Look at the
distribution of your activations. Can you identify "dead neurons" (neurons that output
zero for all inputs)? Compare this run with a Tanh run and explain the difference in
convergence based on the gradients you observed.
"""

import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy_loss
from utils.data_loader import load_dataset, create_batches
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="Dead Neuron Investigation")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_assign1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--high_lr", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    return parser.parse_args()


def _collect_hidden_activations(model, X_batch):
    """Collect hidden layer outputs for one batch."""
    A = X_batch
    activations = {}
    for i, layer in enumerate(model.layers):
        A = layer.forward(A)
        if i < len(model.layers) - 1:
            activations[f"layer_{i}"] = A
    return activations


def _check_plateau(val_history, patience=3, min_delta=1e-3):
    """Simple plateau check: no clear gain in recent epochs."""
    if len(val_history) < patience + 1:
        return False
    recent = val_history[-(patience + 1):]
    return (max(recent) - min(recent)) < min_delta


def train_with_monitoring(activation_name, lr, args):
    run_name = f"dead_neuron_{activation_name}_lr_{lr}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "dataset": args.dataset,
            "activation": activation_name,
            "learning_rate": lr,
            "architecture": [128, 128, 64],
            "optimizer": "sgd",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "loss": "cross_entropy",
            "weight_init": "xavier",
        },
    )

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # Keep last 5000 as validation set.
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]

    model_config = type(
        "Config",
        (),
        {
            "hidden_size": [128, 128, 64],
            "activation": activation_name,
            "weight_init": "xavier",
            "loss": "cross_entropy",
            "optimizer": "sgd",
            "learning_rate": lr,
            "weight_decay": 0.0,
        },
    )()

    model = NeuralNetwork(model_config)
    val_history = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Save per-layer activations across multiple batches.
        hidden_act_store = {f"layer_{i}": [] for i in range(len(model.layers) - 1)}

        first_hidden_grad_norms = []
        mean_hidden_grad_norms = []

        for batch_idx, (X_batch, y_batch) in enumerate(
            create_batches(X_train, y_train, batch_size=args.batch_size, shuffle=True)
        ):
            logits, _ = model.forward(X_batch)
            loss, _ = cross_entropy_loss(y_batch, logits)

            model.backward(y_batch, logits)

            # Gradient norms for convergence comparison.
            first_hidden_grad_norms.append(float(np.linalg.norm(model.layers[0].grad_W)))
            hidden_grad_norms = [
                float(np.linalg.norm(layer.grad_W)) for layer in model.layers[:-1]
            ]
            mean_hidden_grad_norms.append(float(np.mean(hidden_grad_norms)))

            model.update_weights()

            epoch_loss += float(loss)
            num_batches += 1

            # Every 10th batch, capture hidden activations for distribution/dead-neuron checks.
            if batch_idx % 10 == 0:
                batch_acts = _collect_hidden_activations(model, X_batch)
                for layer_name, act in batch_acts.items():
                    hidden_act_store[layer_name].append(act)

        avg_loss = epoch_loss / max(num_batches, 1)

        val_logits, _ = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = float(accuracy_score(val_true, val_preds))
        val_history.append(val_acc)

        test_logits, _ = model.forward(X_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_acc = float(accuracy_score(test_true, test_preds))

        plateau_now = (epoch >= 5) and _check_plateau(val_history, patience=3, min_delta=1e-3)

        log_data = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "first_hidden_grad_norm": float(np.mean(first_hidden_grad_norms)) if first_hidden_grad_norms else 0.0,
            "mean_hidden_grad_norm": float(np.mean(mean_hidden_grad_norms)) if mean_hidden_grad_norms else 0.0,
            "plateau_detected": int(plateau_now),
            "activation": activation_name,
            "learning_rate": lr,
        }

        for layer_name, batch_list in hidden_act_store.items():
            if not batch_list:
                continue

            layer_mat = np.vstack(batch_list)
            flat = layer_mat.reshape(-1)

            zeros_pct = float(np.mean(np.isclose(flat, 0.0)) * 100.0)
            log_data[f"{layer_name}_act_mean"] = float(np.mean(flat))
            log_data[f"{layer_name}_act_std"] = float(np.std(flat))
            log_data[f"{layer_name}_act_min"] = float(np.min(flat))
            log_data[f"{layer_name}_act_max"] = float(np.max(flat))
            log_data[f"{layer_name}_zeros_percent"] = zeros_pct
            log_data[f"{layer_name}_activation_hist"] = wandb.Histogram(flat)

            # Exact dead-neuron count for ReLU: neuron output is ~0 for all monitored samples.
            if activation_name == "relu":
                dead_count = int(np.sum(np.all(np.isclose(layer_mat, 0.0), axis=0)))
                total = int(layer_mat.shape[1])
                log_data[f"{layer_name}_dead_neurons"] = dead_count
                log_data[f"{layer_name}_dead_neuron_percent"] = (100.0 * dead_count / total)

        wandb.log(log_data)

        print(
            f"{activation_name.upper()} | epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f} | "
            f"plateau={plateau_now}"
        )

    wandb.finish()


def main():
    args = parse_args()

    # ReLU with high LR as required, then a Tanh comparison run.
    experiments = [
        ("relu", args.high_lr),
        ("tanh", args.high_lr),
    ]

    print("Starting Dead Neuron Investigation...")
    for activation, lr in experiments:
        print(f"Running {activation.upper()} with lr={lr}")
        train_with_monitoring(activation, lr, args)

    print("Done. Use W&B charts to compare activation histograms, dead-neuron %, and gradient norms.")


if __name__ == "__main__":
    main()
