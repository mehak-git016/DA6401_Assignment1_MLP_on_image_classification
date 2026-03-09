"""
W&B Experiment 2.10: The Fashion-MNIST Transfer Challenge

Run only 3 chosen configurations on Fashion-MNIST and compare transfer from MNIST learnings.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
from sklearn.metrics import accuracy_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, create_batches

SEED = 22


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Transfer Challenge")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_assign1")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--mnist_best_config", type=str, default=None)
    return parser.parse_args()


def _make_model_config(hidden, activation, optimizer, lr):
    return type(
        "Config",
        (),
        {
            "hidden_size": hidden,
            "activation": activation,
            "weight_init": "xavier",
            "loss": "cross_entropy",
            "optimizer": optimizer,
            "learning_rate": lr,
            "weight_decay": 0.0,
        },
    )()


def train_fashion_mnist_config(config_name, cfg, args):
    run_name = f"fashion_mnist_{config_name}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "dataset": "fashion_mnist",
            "config_name": config_name,
            "architecture": cfg["hidden_size"],
            "activation": cfg["activation"],
            "optimizer": cfg["optimizer"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "reasoning": cfg["reasoning"],
            "selection_basis": "chosen from MNIST learnings under 3-config budget",
        },
    )

    np.random.seed(SEED)
    X_train, y_train, X_test, y_test = load_dataset("fashion_mnist")

    model_cfg = _make_model_config(
        cfg["hidden_size"],
        cfg["activation"],
        cfg["optimizer"],
        cfg["learning_rate"],
    )
    model = NeuralNetwork(model_cfg)

    best_test_acc = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        train_preds_all = []
        train_true_all = []

        for X_batch, y_batch in create_batches(
            X_train, y_train, batch_size=args.batch_size, shuffle=True
        ):
            logits, _ = model.forward(X_batch)
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

        test_logits, _ = model.forward(X_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_acc = float(accuracy_score(test_true, test_preds))
        best_test_acc = max(best_test_acc, test_acc)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )

        print(
            f"{config_name} | epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    wandb.summary["best_test_accuracy"] = best_test_acc
    wandb.finish()
    return best_test_acc


def _load_mnist_best_config(path_from_arg):
    if path_from_arg:
        p = Path(path_from_arg)
    else:
        p = Path(__file__).resolve().parent / "best_config.json"

    if not p.exists():
        return None

    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _match_with_mnist_best(best_cfg, candidate_cfg):
    if not best_cfg:
        return False

    # compare only the requested tuple: Architecture + Optimizer + Activation
    best_hidden = best_cfg.get("hidden_size", best_cfg.get("hidden_layers"))
    if isinstance(best_hidden, int):
        best_hidden = [best_hidden]

    return (
        best_hidden == candidate_cfg["hidden_size"]
        and best_cfg.get("optimizer") == candidate_cfg["optimizer"]
        and best_cfg.get("activation") == candidate_cfg["activation"]
    )


def main():
    args = parse_args()

    # Three choices for transfer: include exact MNIST-best tuple + two strong alternatives.
    configurations = {
        "cfg_1_mnist_best_transfer": {
            "hidden_size": [128, 64, 64, 64],
            "activation": "tanh",
            "optimizer": "rmsprop",
            "learning_rate": 0.003110199715274576,
            "reasoning": "Exact Architecture+Optimizer+Activation from MNIST best config.",
        },
        "cfg_2_relu_rmsprop": {
            "hidden_size": [256, 128, 64],
            "activation": "relu",
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "reasoning": "Stronger ReLU baseline with RMSProp for Fashion-MNIST complexity.",
        },
        "cfg_3_relu_nag": {
            "hidden_size": [128, 128, 64],
            "activation": "relu",
            "optimizer": "nag",
            "learning_rate": 0.0005,
            "reasoning": "NAG variant to compare faster directional updates and generalization.",
        },
    }

    print("Starting 2.10 Fashion-MNIST Transfer Challenge")
    print("Running exactly 3 configurations on Fashion-MNIST.")

    results = {}
    for name, cfg in configurations.items():
        print(f"\nRunning {name}: {cfg['hidden_size']} | {cfg['activation']} | {cfg['optimizer']}")
        best_acc = train_fashion_mnist_config(name, cfg, args)
        results[name] = {"best_test_accuracy": best_acc, **cfg}
        print(f"Best test accuracy for {name}: {best_acc:.4f}")

    best_name = max(results.keys(), key=lambda k: results[k]["best_test_accuracy"])
    best_acc = results[best_name]["best_test_accuracy"]

    mnist_best_cfg = _load_mnist_best_config(args.mnist_best_config)
    same_as_mnist_best = _match_with_mnist_best(mnist_best_cfg, results[best_name])

    # Summary run for one-place reporting.
    wandb.init(project=args.wandb_project, name="fashion_mnist_transfer_summary")

    table = wandb.Table(
        columns=[
            "config_name",
            "architecture",
            "activation",
            "optimizer",
            "learning_rate",
            "best_test_accuracy",
            "reasoning",
        ]
    )
    for name, r in results.items():
        table.add_data(
            name,
            str(r["hidden_size"]),
            r["activation"],
            r["optimizer"],
            r["learning_rate"],
            r["best_test_accuracy"],
            r["reasoning"],
        )

    interpretation = (
        "Fashion-MNIST has higher intra-class variation and inter-class similarity than digits, "
        "so some MNIST-best settings may not transfer as-is; capacity/optimization trade-offs matter more."
    )

    wandb.log({
        "fashion_mnist_results": table,
        "best_config_name": best_name,
        "best_accuracy": best_acc,
        "mnist_best_also_fashion_best": int(same_as_mnist_best),
    })

    wandb.summary["transfer_interpretation"] = interpretation

    print("\n" + "=" * 60)
    print("2.10 RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(
            f"{name}: acc={r['best_test_accuracy']:.4f} | "
            f"arch={r['hidden_size']} | act={r['activation']} | opt={r['optimizer']}"
        )

    print(f"\nBest on Fashion-MNIST: {best_name} ({best_acc:.4f})")
    if mnist_best_cfg is None:
        print("MNIST best config file not found; direct best-vs-best transfer comparison skipped.")
    else:
        print(f"Did MNIST best also win on Fashion-MNIST? {'Yes' if same_as_mnist_best else 'No'}")

    print("Interpretation:")
    print(interpretation)

    wandb.finish()


if __name__ == "__main__":
    main()
