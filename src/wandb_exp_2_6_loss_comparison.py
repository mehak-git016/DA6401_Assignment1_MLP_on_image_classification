"""
W&B Experiment 2.6: Loss Function Comparison

Compare the training curves of two models: one using Mean Squared Error (MSE)
and one using Cross-Entropy. Use the same architecture and learning rate for both.
Which loss function converged faster? Theoretically, why is Cross-Entropy better suited
for multi-class classification when paired with a Softmax output?
"""

import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, create_batches
from sklearn.metrics import accuracy_score

SEED = 22


def parse_args():
    parser = argparse.ArgumentParser(description="Loss Function Comparison")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_assign1")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=15)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    return parser.parse_args()


def train_with_loss(loss_name, args):
    """Train one run for the given loss with the same architecture/hyperparams."""

    # Reset seed so both loss runs start from comparable initialization/data order.
    np.random.seed(SEED)

    run_name = f"loss_comparison_{loss_name}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "dataset": args.dataset,
            "loss_function": loss_name,
            "architecture": [128, 64],
            "activation": "relu",
            "optimizer": "rmsprop",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_init": "xavier",
            "seed": SEED,
        },
    )

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # Simple validation split from training data.
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]

    model_config = type(
        "Config",
        (),
        {
            "hidden_size": [128, 64],
            "activation": "relu",
            "weight_init": "xavier",
            "loss": loss_name,
            "optimizer": "rmsprop",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
        },
    )()

    model = NeuralNetwork(model_config)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
    }

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        train_preds_all = []
        train_true_all = []

        for X_batch, y_batch in create_batches(
            X_train, y_train, batch_size=args.batch_size, shuffle=True
        ):
            logits, _ = model.forward(X_batch)

            # Use model's configured loss so CE vs MSE is the only changing factor.
            loss, _ = model.loss_fn(y_batch, logits)
            model.backward(y_batch, logits)
            model.update_weights()

            epoch_loss += float(loss)
            num_batches += 1

            train_preds_all.append(np.argmax(logits, axis=1))
            train_true_all.append(np.argmax(y_batch, axis=1))

        avg_loss = epoch_loss / max(num_batches, 1)

        train_preds = np.concatenate(train_preds_all)
        train_true = np.concatenate(train_true_all)
        train_acc = float(accuracy_score(train_true, train_preds))

        val_logits, _ = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = float(accuracy_score(val_true, val_preds))

        test_logits, _ = model.forward(X_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_acc = float(accuracy_score(test_true, test_preds))

        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["test_accuracy"].append(test_acc)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "loss_function": loss_name,
            }
        )

        print(
            f"{loss_name.upper()} | epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
        )

    wandb.finish()
    return history


def main():
    args = parse_args()

    print("Starting Loss Function Comparison...")
    print("Same architecture, same learning rate, only loss changes.")

    ce_hist = train_with_loss("cross_entropy", args)
    mse_hist = train_with_loss("mse", args)

    # Simple convergence summary based on final epoch training loss.
    ce_final = ce_hist["train_loss"][-1]
    mse_final = mse_hist["train_loss"][-1]
    faster = "cross_entropy" if ce_final < mse_final else "mse"

    print("\nComparison finished.")
    print(f"Lower final training loss: {faster}")
    print("In general, cross-entropy works better for multi-class classification with softmax")
    print("because it gives stronger probability-aware gradients than MSE.")


if __name__ == "__main__":
    main()
