"""
W&B Experiment 2.9: Weight Initialization & Symmetry

Compare Zeros vs Xavier initialization by tracking gradients of
5 neurons from the same hidden layer over the first 50 iterations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, create_batches

SEED = 22


def parse_args():
    parser = argparse.ArgumentParser(description="Weight Initialization & Symmetry")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_assign1")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def _build_model(init_type, lr):
    model_config = type(
        "Config",
        (),
        {
            "hidden_size": [128, 64],
            "activation": "relu",
            "weight_init": "xavier",
            "loss": "cross_entropy",
            "optimizer": "sgd",
            "learning_rate": lr,
            "weight_decay": 0.0,
        },
    )()

    model = NeuralNetwork(model_config)

    if init_type == "zeros":
        for layer in model.layers:
            layer.W = np.zeros_like(layer.W)
            layer.b = np.zeros_like(layer.b)

    return model


def _track_gradients(init_type, args):
    np.random.seed(SEED)

    X_train, y_train, _, _ = load_dataset(args.dataset)

    # Small subset is enough for first 50 iterations tracking.
    subset_n = min(len(X_train), args.batch_size * (args.iterations + 2))
    X_train = X_train[:subset_n]
    y_train = y_train[:subset_n]

    model = _build_model(init_type, args.learning_rate)

    grad_traces = {f"neuron_{i}": [] for i in range(5)}
    losses = []
    steps = []

    step = 0
    for X_batch, y_batch in create_batches(
        X_train, y_train, batch_size=args.batch_size, shuffle=False
    ):
        if step >= args.iterations:
            break

        logits, _ = model.forward(X_batch)
        loss, _ = model.loss_fn(y_batch, logits)

        # Use the y_true,y_pred path for compatibility.
        model.backward(y_batch, logits)

        # First hidden layer gradients: (input_dim, hidden_units).
        g = model.layers[0].grad_W

        for i in range(5):
            if i < g.shape[1]:
                # Mean abs gradient for one neuron's incoming weights.
                grad_traces[f"neuron_{i}"].append(float(np.mean(np.abs(g[:, i]))))

        losses.append(float(loss))
        steps.append(step)

        model.update_weights()
        step += 1

    # Symmetry score: std across the 5 neurons at each step.
    grad_matrix = np.array([grad_traces[f"neuron_{i}"] for i in range(5)])
    symmetry_std = np.std(grad_matrix, axis=0) if grad_matrix.size else np.array([])

    return {
        "steps": steps,
        "grad_traces": grad_traces,
        "losses": losses,
        "symmetry_std": symmetry_std.tolist(),
    }


def _plot_run(tracking, init_type):
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(5):
        neuron = f"neuron_{i}"
        ax.plot(tracking["steps"], tracking["grad_traces"][neuron], label=neuron, linewidth=2)

    ax.set_title(f"{init_type.upper()} init: 5-neuron gradient traces (first hidden layer)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean |gradient|")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def _plot_combined(zeros_tracking, xavier_tracking):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i in range(5):
        neuron = f"neuron_{i}"
        axes[0].plot(zeros_tracking["steps"], zeros_tracking["grad_traces"][neuron], label=neuron, linewidth=2)
        axes[1].plot(xavier_tracking["steps"], xavier_tracking["grad_traces"][neuron], label=neuron, linewidth=2)

    axes[0].set_title("Zeros Initialization")
    axes[1].set_title("Xavier Initialization")

    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean |gradient| (first hidden layer)")
    axes[1].legend(loc="best")

    fig.suptitle("2.9 Symmetry Breaking: 5-Neuron Gradient Lines over First 50 Iterations")
    return fig


def main():
    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        name="2.9_weight_init_symmetry",
        config={
            "dataset": args.dataset,
            "architecture": [128, 64],
            "activation": "relu",
            "optimizer": "sgd",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "tracked_iterations": args.iterations,
            "compared_inits": ["zeros", "xavier"],
            "loss": "cross_entropy",
        },
    )

    print("Tracking gradients for ZEROS initialization...")
    zeros_tracking = _track_gradients("zeros", args)

    print("Tracking gradients for XAVIER initialization...")
    xavier_tracking = _track_gradients("xavier", args)

    zeros_fig = _plot_run(zeros_tracking, "zeros")
    xavier_fig = _plot_run(xavier_tracking, "xavier")
    combined_fig = _plot_combined(zeros_tracking, xavier_tracking)

    # Quantify overlap of neuron lines.
    zeros_sym = float(np.mean(zeros_tracking["symmetry_std"])) if zeros_tracking["symmetry_std"] else 0.0
    xavier_sym = float(np.mean(xavier_tracking["symmetry_std"])) if xavier_tracking["symmetry_std"] else 0.0

    wandb.log(
        {
            "zeros_gradient_plot": wandb.Image(zeros_fig),
            "xavier_gradient_plot": wandb.Image(xavier_fig),
            "combined_gradient_comparison": wandb.Image(combined_fig),
            "zeros_mean_neuron_gradient_std": zeros_sym,
            "xavier_mean_neuron_gradient_std": xavier_sym,
        }
    )

    # Save concise interpretation in summary.
    wandb.summary["symmetry_observation"] = (
        "With zeros init, neuron gradient lines overlap because neurons start identical and receive identical updates."
    )
    wandb.summary["why_symmetry_breaking_needed"] = (
        "Different initial weights create different gradients, so neurons specialize into distinct features."
    )

    print(f"Zeros symmetry score (lower means more overlap): {zeros_sym:.8f}")
    print(f"Xavier symmetry score: {xavier_sym:.8f}")
    print("2.9 experiment complete.")

    plt.close(zeros_fig)
    plt.close(xavier_fig)
    plt.close(combined_fig)

    wandb.finish()


if __name__ == "__main__":
    main()
