"""Data loading and manipulation utilities for MNIST experiments."""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset


def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def load_mnist(device=None, train=True):
    """
    Load MNIST dataset with standard normalization.

    Args:
        device: torch device to load data onto. If None, uses get_device().
        train: If True, load training set (60k). If False, load test set (10k).

    Returns:
        images: Tensor of shape (N, 1, 28, 28), normalized
        labels: Tensor of shape (N,)
    """
    if device is None:
        device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data', train=train, download=True, transform=transform
    )

    # Extract into tensors with normalization
    images = dataset.data.float().unsqueeze(1) / 255.0
    images = (images - 0.1307) / 0.3081
    labels = dataset.targets

    images = images.to(device)
    labels = labels.to(device)

    return images, labels


def create_clusters(num_samples, cluster_sizes, seed=None):
    """
    Create disjoint clusters of data indices.

    Uses a single random permutation to ensure clusters don't overlap
    across different sizes.

    Args:
        num_samples: Total number of samples in dataset
        cluster_sizes: List of cluster sizes to create
        seed: Random seed for reproducibility

    Returns:
        dict mapping cluster_size -> list of index arrays
    """
    if seed is not None:
        np.random.seed(seed)

    permutation = np.random.permutation(num_samples)

    all_clusters = {}
    for size in cluster_sizes:
        num_clusters = num_samples // size
        clusters = []
        for i in range(num_clusters):
            start = i * size
            end = start + size
            clusters.append(permutation[start:end])
        all_clusters[size] = clusters

    return all_clusters


def create_digit_subset(images, labels, digits):
    """
    Filter dataset to only include specific digit classes.

    Args:
        images: Tensor of images
        labels: Tensor of labels
        digits: List of digit classes to keep (e.g., [0, 1, 2])

    Returns:
        filtered_images, filtered_labels, original_indices
    """
    mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    for d in digits:
        mask |= (labels == d)

    indices = torch.where(mask)[0]
    return images[indices], labels[indices], indices


def create_data_loader(images, labels, indices, batch_size=128, shuffle=True):
    """
    Create a DataLoader for a subset of data.

    Args:
        images: Full image tensor
        labels: Full label tensor
        indices: Indices to include in this loader
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    X = images[indices]
    y = labels[indices]
    dataset = TensorDataset(X, y)
    actual_batch_size = min(batch_size, len(indices))

    return DataLoader(dataset, batch_size=actual_batch_size, shuffle=shuffle, drop_last=False)
