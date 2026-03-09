"""
W&B Experiment 2.7: Global Performance Analysis

Create an overlay plot showing Training vs Test Accuracy across every run in a sweep.
Also identify runs with high training accuracy but poor test accuracy.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb

PROJECT = "DA6401_assign1"


def parse_args():
    parser = argparse.ArgumentParser(description="Global Performance Analysis")
    parser.add_argument("-wp", "--wandb_project", "--project", dest="project", type=str, default=PROJECT)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--gap_threshold", type=float, default=0.08)
    parser.add_argument("--high_train_threshold", type=float, default=0.95)
    return parser.parse_args()


def _get_metric_from_run(run, key):
    """Read metric from summary, fallback to history if needed."""
    summary = run.summary
    if key in summary and summary[key] is not None:
        return float(summary[key])

    # fallback for runs where summary was not finalized properly
    try:
        hist = run.history(keys=[key], pandas=False)
        vals = [row.get(key) for row in hist if row.get(key) is not None]
        if vals:
            return float(vals[-1])
    except Exception:
        pass

    return None


def main():
    args = parse_args()

    sweep_id = args.sweep_id
    if not sweep_id:
        sweep_id = input("Enter the Sweep ID from Task 2.2: ").strip()

    wandb.init(project=args.project, name="global_performance_analysis")

    api = wandb.Api()
    entity = wandb.Api().default_entity
    sweep = api.sweep(f"{entity}/{args.project}/{sweep_id}")
    runs = sweep.runs

    rows = []
    for run in runs:
        tr = _get_metric_from_run(run, "train_accuracy")
        te = _get_metric_from_run(run, "test_accuracy")
        if tr is None or te is None:
            continue

        gap = tr - te
        rows.append(
            {
                "run_name": run.name,
                "run_id": run.id,
                "train_accuracy": tr,
                "test_accuracy": te,
                "gap": gap,
                "optimizer": run.config.get("optimizer", "na"),
                "learning_rate": run.config.get("learning_rate", "na"),
                "activation": run.config.get("activation", "na"),
            }
        )

    if not rows:
        print("No runs with both train_accuracy and test_accuracy were found.")
        wandb.finish()
        return

    # Sort for cleaner overlay visualization.
    rows.sort(key=lambda x: x["test_accuracy"], reverse=True)

    run_idx = np.arange(len(rows))
    train_acc = np.array([r["train_accuracy"] for r in rows])
    test_acc = np.array([r["test_accuracy"] for r in rows])
    gaps = np.array([r["gap"] for r in rows])

    # High-train but poor-test criteria.
    overfit_mask = (train_acc >= args.high_train_threshold) & (gaps >= args.gap_threshold)
    overfit_indices = np.where(overfit_mask)[0]

    print(f"Total runs analysed: {len(rows)}")
    print(f"High-train/poor-test runs: {len(overfit_indices)}")

    if len(overfit_indices) > 0:
        print("\nRuns with high train acc but poor test acc:")
        for i in overfit_indices:
            r = rows[i]
            print(
                f"- {r['run_name']} ({r['run_id']}): "
                f"train={r['train_accuracy']:.4f}, test={r['test_accuracy']:.4f}, "
                f"gap={r['gap']:.4f}, opt={r['optimizer']}, lr={r['learning_rate']}, act={r['activation']}"
            )

    # Figure 1: overlay lines of train vs test across all runs.
    plt.figure(figsize=(11, 6))
    plt.plot(run_idx, train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(run_idx, test_acc, label="Test Accuracy", linewidth=2)

    if len(overfit_indices) > 0:
        plt.scatter(overfit_indices, train_acc[overfit_indices], marker="o", s=60, label="High-train points")
        plt.scatter(overfit_indices, test_acc[overfit_indices], marker="x", s=70, label="Poor-test points")

    plt.xlabel("Run index (sorted by test accuracy)")
    plt.ylabel("Accuracy")
    plt.title("2.7 Global Performance: Train vs Test Accuracy Across Sweep Runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    wandb.log({"train_vs_test_accuracy_overlay": wandb.Image(plt)})

    # Figure 2: scatter + train=test reference for quick generalization view.
    plt.figure(figsize=(7, 7))
    plt.scatter(train_acc, test_acc, alpha=0.75, label="Runs")
    if len(overfit_indices) > 0:
        plt.scatter(
            train_acc[overfit_indices],
            test_acc[overfit_indices],
            marker="x",
            s=90,
            label="High train, poor test",
        )

    m1 = float(min(train_acc.min(), test_acc.min()))
    m2 = float(max(train_acc.max(), test_acc.max()))
    plt.plot([m1, m2], [m1, m2], "--", label="Train = Test")
    plt.xlabel("Training Accuracy")
    plt.ylabel("Test Accuracy")
    plt.title("Generalization Gap View")
    plt.grid(True, alpha=0.3)
    plt.legend()
    wandb.log({"train_test_scatter_with_gap": wandb.Image(plt)})

    # Log a table with the identified runs so the answer is explicit.
    table = wandb.Table(columns=["run_name", "run_id", "train_accuracy", "test_accuracy", "gap", "optimizer", "learning_rate", "activation"])
    for r in rows:
        table.add_data(
            r["run_name"],
            r["run_id"],
            r["train_accuracy"],
            r["test_accuracy"],
            r["gap"],
            r["optimizer"],
            r["learning_rate"],
            r["activation"],
        )
    wandb.log({"all_runs_accuracy_table": table})

    # Clear textual interpretation requested in objective.
    if len(overfit_indices) > 0:
        interpretation = (
            "Large train-test gap indicates overfitting: model memorizes training patterns "
            "but does not generalize well to unseen data."
        )
    else:
        interpretation = "No major overfitting runs under current thresholds."

    wandb.summary["overfitting_interpretation"] = interpretation
    print("\nInterpretation:")
    print(interpretation)

    wandb.finish()


if __name__ == "__main__":
    main()
