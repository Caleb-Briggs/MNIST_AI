"""Evaluation utilities for MNIST experiments."""

import torch
import numpy as np


@torch.no_grad()
def evaluate(model, images, labels, indices=None, batch_size=2048):
    """
    Evaluate a model's accuracy on given data.

    Args:
        model: PyTorch model
        images: Image tensor
        labels: Label tensor
        indices: Optional indices to evaluate on (if None, uses all data)
        batch_size: Batch size for evaluation

    Returns:
        accuracy (float between 0 and 1)
    """
    model.eval()

    if indices is not None:
        X = images[indices]
        y = labels[indices]
    else:
        X = images
        y = labels

    correct = 0
    total = 0

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        output = model(batch_X)
        preds = output.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return correct / total


@torch.no_grad()
def cross_evaluate(models, images, labels, clusters, train_cluster_indices):
    """
    Evaluate multiple models on multiple clusters.

    Creates an MxM accuracy matrix where entry [i,j] is model i's accuracy
    on cluster j's data. Diagonal entries are NaN (self-evaluation excluded).

    Args:
        models: List of M trained models
        images: Full image tensor
        labels: Full label tensor
        clusters: List of all cluster index arrays
        train_cluster_indices: List of M cluster indices that each model was trained on

    Returns:
        accuracy_matrix: (M, M) numpy array
    """
    num_models = len(models)
    accuracy_matrix = np.full((num_models, num_models), np.nan)

    for m in models:
        m.eval()

    # Cache cluster data
    cluster_data = []
    for c_idx in train_cluster_indices:
        c_indices = clusters[c_idx]
        X = images[c_indices]
        y = labels[c_indices]
        cluster_data.append((X, y))

    # Evaluate: model i on cluster j
    for j, (X, y) in enumerate(cluster_data):
        for i, model in enumerate(models):
            if i == j:  # Skip self-evaluation
                continue

            output = model(X)
            preds = output.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            accuracy_matrix[i, j] = acc

    return accuracy_matrix


def compute_stats(matrix, exclude_diagonal=True):
    """
    Compute statistics from an accuracy matrix.

    Args:
        matrix: NxM numpy array (may contain NaN for diagonal)
        exclude_diagonal: If True, excludes diagonal entries

    Returns:
        dict with mean, std, median, min, max, percentiles
    """
    if exclude_diagonal:
        values = matrix[~np.isnan(matrix)]
    else:
        values = matrix.flatten()

    return {
        'mean': values.mean(),
        'std': values.std(),
        'median': np.median(values),
        'min': values.min(),
        'max': values.max(),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
    }


def generalization_gap(train_acc, test_acc):
    """
    Compute generalization gap.

    Args:
        train_acc: Training accuracy (or array of accuracies)
        test_acc: Test accuracy (or array of accuracies)

    Returns:
        train_acc - test_acc
    """
    return np.array(train_acc) - np.array(test_acc)


@torch.no_grad()
def get_predictions(model, images, labels, indices=None, batch_size=2048):
    """
    Get model predictions and correctness for analysis.

    Args:
        model: PyTorch model
        images: Image tensor
        labels: Label tensor
        indices: Optional indices (if None, uses all)
        batch_size: Batch size

    Returns:
        dict with predictions, true_labels, correct (boolean), confidences
    """
    model.eval()

    if indices is not None:
        X = images[indices]
        y = labels[indices]
    else:
        X = images
        y = labels

    all_preds = []
    all_probs = []

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        output = model(batch_X)
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    y_cpu = y.cpu()

    return {
        'predictions': preds.numpy(),
        'true_labels': y_cpu.numpy(),
        'correct': (preds == y_cpu).numpy(),
        'confidences': probs.max(dim=1).values.numpy(),
        'probabilities': probs.numpy(),
    }
