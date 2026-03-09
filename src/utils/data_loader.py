"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets using keras.datasets
"""

import numpy as np


def load_dataset(dataset_name='mnist'):
    """
    Load MNIST or Fashion-MNIST using keras.datasets and preprocess.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'

    Returns:
        X_train: Training images, shape (~60000, 784), normalized to [0,1]
        y_train: Training labels, one-hot encoded
        X_test: Test images, shape (~10000, 784), normalized to [0,1]
        y_test: Test labels, one-hot encoded
    """

    if dataset_name not in ('mnist', 'fashion_mnist'):
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")

    if dataset_name == 'mnist':
        from keras.datasets import mnist as keras_ds
    else:
        from keras.datasets import fashion_mnist as keras_ds

    (X_train, y_train), (X_test, y_test) = keras_ds.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

    # one-hot encoding
    num_classes = 10
    y_train = one_hot_encode(y_train, num_classes)
    y_test = one_hot_encode(y_test, num_classes)

    return X_train, y_train, X_test, y_test


def one_hot_encode(labels, num_classes):
    """
    Convert class labels to one-hot encoded vectors.
    """

    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)

    for i, label in enumerate(labels):
        one_hot[i, label] = 1

    return one_hot


def create_batches(X, y, batch_size=32, shuffle=True):
    """
    Create mini-batches from data for training.
    """

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):

        end_idx = min(start_idx + batch_size, num_samples)

        batch_indices = indices[start_idx:end_idx]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        yield X_batch, y_batch
