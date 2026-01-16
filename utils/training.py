"""Training utilities for MNIST experiments."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, images, labels, indices, epochs=20, lr=1e-3, weight_decay=1e-4, batch_size=128):
    """
    Train a single model on given data indices.

    Args:
        model: PyTorch model (should already be on device)
        images: Full image tensor (on device)
        labels: Full label tensor (on device)
        indices: Indices of samples to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        batch_size: Batch size

    Returns:
        dict with training history (losses)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    X = images[indices]
    y = labels[indices]

    actual_batch_size = min(batch_size, len(indices))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)

    history = {'loss': []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        history['loss'].append(epoch_loss / len(loader))

    return history


def train_parallel(models, images, labels, list_of_indices, epochs=20, lr=1e-3, weight_decay=1e-4):
    """
    Train multiple models in parallel by interleaving gradient updates.

    More efficient for small training sets where each model's data fits in memory.

    Args:
        models: List of PyTorch models (should already be on device)
        images: Full image tensor (on device)
        labels: Full label tensor (on device)
        list_of_indices: List of index arrays, one per model
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW

    Returns:
        List of trained models (same as input, modified in place)
    """
    num_models = len(models)

    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for m in models
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for opt in optimizers
    ]

    # Prepare data for each model
    data_X = [images[idx] for idx in list_of_indices]
    data_y = [labels[idx] for idx in list_of_indices]

    for epoch in range(epochs):
        for m in models:
            m.train()

        # For small clusters, do one "batch" per epoch (the whole cluster)
        for i in range(num_models):
            X, y = data_X[i], data_y[i]

            optimizers[i].zero_grad()
            output = models[i](X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizers[i].step()

        for s in schedulers:
            s.step()

    return models


def create_optimizer(model, lr=1e-3, weight_decay=1e-4, optimizer_type='adamw'):
    """
    Create an optimizer for a model.

    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_type: 'adamw', 'adam', or 'sgd'

    Returns:
        optimizer
    """
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
