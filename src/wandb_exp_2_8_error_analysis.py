"""
W&B Experiment 2.8: Error Analysis

Plot a confusion matrix for the best model on the test set.
Also add a creative failure view: top confusion pairs + sample mistakes.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Error Analysis Experiment")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_assign1")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    return parser.parse_args()


def _default_paths(base_dir: Path, model_path: str | None, config_path: str | None):
    model = Path(model_path) if model_path else (base_dir / "best_model.npy")
    config = Path(config_path) if config_path else (base_dir / "best_config.json")
    return model, config


def _to_model_config(cfg: dict):
    # Keep this mapping flexible because sweep/config keys may vary.
    hidden = cfg.get("hidden_size", cfg.get("hidden_layers", [128, 64]))
    if isinstance(hidden, int):
        hidden = [hidden]

    return type(
        "Config",
        (),
        {
            "hidden_size": hidden,
            "activation": cfg.get("activation", "relu"),
            "weight_init": cfg.get("weight_init", "xavier"),
            "loss": cfg.get("loss", "cross_entropy"),
            "optimizer": cfg.get("optimizer", "rmsprop"),
            "learning_rate": float(cfg.get("learning_rate", 0.001)),
            "weight_decay": float(cfg.get("weight_decay", 0.0)),
        },
    )()


def load_best_model(model_path: Path, config_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"best model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"best config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = _to_model_config(cfg)
    model = NeuralNetwork(model_cfg)

    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    return model, cfg


def create_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = list(range(10))
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Best Model)")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)

    return fig, cm


def create_top_confusions_plot(cm):
    # Ignore diagonal; keep only mistakes.
    cm_off = cm.copy().astype(float)
    np.fill_diagonal(cm_off, 0)

    pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm_off[t, p] > 0:
                pairs.append((cm_off[t, p], t, p))

    pairs.sort(reverse=True)
    top = pairs[:8]

    fig, ax = plt.subplots(figsize=(10, 5))
    if not top:
        ax.text(0.5, 0.5, "No misclassifications found", ha="center", va="center")
        ax.axis("off")
        return fig, []

    labels = [f"{t}->{p}" for _, t, p in top]
    values = [int(c) for c, _, _ in top]
    ax.bar(labels, values)
    ax.set_title("Top Confusion Pairs (True -> Pred)")
    ax.set_xlabel("Class Pair")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    return fig, top


def create_failure_gallery(X_test, y_true, y_pred, top_pairs, max_cols=5):
    # Show real failed examples from the worst confusion pairs.
    chosen_indices = []
    for _, t, p in top_pairs[:4]:
        idx = np.where((y_true == t) & (y_pred == p))[0]
        if len(idx) > 0:
            chosen_indices.extend(list(idx[:2]))

    chosen_indices = chosen_indices[: max_cols * 2]
    if not chosen_indices:
        return None

    rows = 2
    cols = max_cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 4.8))
    axes = axes.flatten()

    for i in range(rows * cols):
        axes[i].axis("off")

    for i, idx in enumerate(chosen_indices):
        img = X_test[idx].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"T:{y_true[idx]} P:{y_pred[idx]}", color="red", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("Failure Gallery: Real Misclassified Test Samples", fontsize=12)
    return fig


def main():
    args = parse_args()

    src_dir = Path(__file__).resolve().parent
    model_path, config_path = _default_paths(src_dir, args.model_path, args.config_path)

    wandb.init(project=args.wandb_project, name="2.8_error_analysis")

    model, used_cfg = load_best_model(model_path, config_path)

    _, _, X_test, y_test = load_dataset(args.dataset)
    logits, _ = model.forward(X_test)

    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)

    cm_fig, cm = create_confusion_matrix_plot(y_true, y_pred)
    top_conf_fig, top_confusions = create_top_confusions_plot(cm)
    gallery_fig = create_failure_gallery(X_test, y_true, y_pred, top_confusions)

    logs = {
        "test_accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "confusion_matrix": wandb.Image(cm_fig),
        "top_confusions": wandb.Image(top_conf_fig),
    }
    if gallery_fig is not None:
        logs["failure_gallery"] = wandb.Image(gallery_fig)

    wandb.log(logs)

    # Log per-class metrics in a table.
    class_table = []
    for c in range(10):
        c_key = str(c)
        class_table.append(
            [
                c,
                report[c_key]["precision"],
                report[c_key]["recall"],
                report[c_key]["f1-score"],
                report[c_key]["support"],
            ]
        )

    wandb.log(
        {
            "classification_report": wandb.Table(
                columns=["Class", "Precision", "Recall", "F1", "Support"],
                data=class_table,
            )
        }
    )

    # Keep key info in run summary for quick grading read.
    wandb.summary["used_model_path"] = str(model_path)
    wandb.summary["used_config_path"] = str(config_path)
    wandb.summary["used_best_config"] = json.dumps(used_cfg)

    if top_confusions:
        top_count, t, p = top_confusions[0]
        wandb.summary["worst_confusion_pair"] = f"{t}->{p} ({int(top_count)})"

    print(f"Test accuracy: {accuracy:.4f}")
    print("2.8 analysis complete.")

    plt.close(cm_fig)
    plt.close(top_conf_fig)
    if gallery_fig is not None:
        plt.close(gallery_fig)

    wandb.finish()


if __name__ == "__main__":
    main()
