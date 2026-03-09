"""
W&B Experiments Master Script

Run all W&B experiments for the assignment. Each experiment is in a separate file.

Usage:
    python wandb_experiments.py --experiment 2.1 --project DA6401_assign1
    python wandb_experiments.py --all --project DA6401_assign1

Available experiments:
    2.1: Data Exploration and Class Distribution
    2.2: Hyperparameter Sweep
    2.3: The Optimizer Showdown
    2.4: Vanishing Gradient Analysis
    2.5: The "Dead Neuron" Investigation
    2.6: Loss Function Comparison
    2.7: Global Performance Analysis
    2.8: Error Analysis
    2.9: Weight Initialization & Symmetry
    2.10: The Fashion-MNIST Transfer Challenge
"""

import subprocess
import sys
import argparse


def run_experiment(experiment_num, project_name):
    """Run a specific experiment"""

    script_map = {
        "2.1": "wandb_exp_2_1_data_exploration.py",
        "2.2": "wandb_exp_2_2_hyperparameter_sweep.py",
        "2.3": "wandb_exp_2_3_optimizer_showdown.py",
        "2.4": "wandb_exp_2_4_vanishing_gradient.py",
        "2.5": "wandb_exp_2_5_dead_neurons.py",
        "2.6": "wandb_exp_2_6_loss_comparison.py",
        "2.7": "wandb_exp_2_7_global_performance.py",
        "2.8": "wandb_exp_2_8_error_analysis.py",
        "2.9": "wandb_exp_2_9_weight_init.py",
        "2.10": "wandb_exp_2_10_fashion_mnist.py"
    }

    if experiment_num not in script_map:
        print(f"Unknown experiment: {experiment_num}")
        return False

    script_name = script_map[experiment_num]

    print(f"Running Experiment {experiment_num}: {script_name}")
    print(f"Project: {project_name}")

    # Run the experiment
    cmd = [sys.executable, script_name, "-wp", project_name]
    result = subprocess.run(cmd, cwd=".")

    if result.returncode == 0:
        print(f"Experiment {experiment_num} completed successfully!")
        return True
    else:
        print(f"Experiment {experiment_num} failed!")
        return False


def run_all_experiments(project_name):
    """Run all experiments in sequence"""

    experiments = ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10"]

    print("Running ALL W&B Experiments...")
    print(f"Project: {project_name}")

    successful = 0
    total = len(experiments)

    for exp_num in experiments:
        success = run_experiment(exp_num, project_name)
        if success:
            successful += 1
        print()

    print("="*60)
    print(f"EXPERIMENTS COMPLETED: {successful}/{total} successful")
    print("="*60)

    if successful == total:
        print("All experiments completed successfully!")
        print("Check W&B dashboard for all results")
    else:
        print(f"{total - successful} experiments failed. Check logs above.")


def main():
    parser = argparse.ArgumentParser(description='W&B Experiments Master Script')
    parser.add_argument('--experiment', '-e', type=str,
                        help='Specific experiment to run (e.g., 2.1, 2.2, etc.)')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--project', '-p', type=str, required=True,
                        help='W&B project name')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')

    args = parser.parse_args()

    if args.list:
        print("Available W&B Experiments:")
        print("  2.1: Data Exploration and Class Distribution")
        print("  2.2: Hyperparameter Sweep")
        print("  2.3: The Optimizer Showdown")
        print("  2.4: Vanishing Gradient Analysis")
        print("  2.5: The Dead Neuron Investigation")
        print("  2.6: Loss Function Comparison")
        print("  2.7: Global Performance Analysis")
        print("  2.8: Error Analysis")
        print("  2.9: Weight Initialization & Symmetry")
        print("  2.10: The Fashion-MNIST Transfer Challenge")
        return

    if not args.experiment and not args.all:
        print("Please specify --experiment <num> or --all")
        parser.print_help()
        return

    if args.all:
        run_all_experiments(args.project)
    else:
        run_experiment(args.experiment, args.project)


if __name__ == '__main__':
    main()