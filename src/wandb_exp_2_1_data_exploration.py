"""
W&B Experiment 2.1: Data Exploration and Class Distribution

Log a W&B Table containing 5 sample images from each of the 10 classes in the dataset.
Identify any classes that look visually similar in their raw form. How might this visual
similarity impact your model?
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Data Exploration Experiment')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_assign1')
    return parser.parse_args()


def create_sample_table(X_train, y_train, dataset_name):
    """Create W&B table with 5 samples from each class"""

    # Get class names
    if dataset_name == 'mnist':
        class_names = [str(i) for i in range(10)]
    else:  # fashion_mnist
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Create table columns
    columns = ["Class", "Class Name", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"]

    # Initialize table data
    table_data = []

    # For each class, get 5 samples
    for class_idx in range(10):
        # Find indices of this class
        class_indices = np.where(np.argmax(y_train, axis=1) == class_idx)[0]

        # Take first 5 samples
        sample_indices = class_indices[:5]

        # Create row data
        row = [class_idx, class_names[class_idx]]

        # Add images
        for idx in sample_indices:
            # Reshape to 28x28 and create image
            img = X_train[idx].reshape(28, 28)
            # Convert to wandb.Image
            wandb_img = wandb.Image(img, caption=f"Class {class_idx}")
            row.append(wandb_img)

        table_data.append(row)

    # Create wandb table
    table = wandb.Table(data=table_data, columns=columns)

    return table, class_names


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"2.1_data_exploration_{args.dataset}",
        config={"experiment": "2.1", "dataset": args.dataset}
    )

    print(f"Loading {args.dataset} dataset...")
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    print("Creating sample table...")
    sample_table, class_names = create_sample_table(X_train, y_train, args.dataset)

    # Log the table
    wandb.log({"sample_images_table": sample_table})

    # Log class distribution
    class_counts = np.sum(y_train, axis=0)
    class_dist = {class_names[i]: int(class_counts[i]) for i in range(10)}

    wandb.log({"class_distribution": wandb.Table(
        data=[[k, v] for k, v in class_dist.items()],
        columns=["Class", "Count"]
    )})

    print("Data exploration completed!")
    print(f"Logged to W&B: {wandb.run.url}")

    wandb.finish()


if __name__ == '__main__':
    main()