"""Model architectures for MNIST experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Small CNN for MNIST (~100k parameters).

    Architecture:
    - 3 conv layers (32, 64, 128 channels) with batch norm and max pooling
    - 2 FC layers (256 hidden units)
    - Dropout for regularization
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TinyCNN(nn.Module):
    """
    Tiny CNN for MNIST (~10k parameters).

    Useful for quick experiments or studying small-capacity models.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    """
    Simple MLP for MNIST.

    Args:
        hidden_sizes: List of hidden layer sizes (e.g., [256, 128])
        num_classes: Number of output classes
    """

    def __init__(self, hidden_sizes=[256, 128], num_classes=10):
        super().__init__()
        layers = []
        in_size = 28 * 28

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h

        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.net(x)


def create_model(name, device=None, **kwargs):
    """
    Factory function to create models by name.

    Args:
        name: Model name ('small_cnn', 'tiny_cnn', 'mlp')
        device: Device to place model on
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model on the specified device
    """
    models = {
        'small_cnn': SmallCNN,
        'tiny_cnn': TinyCNN,
        'mlp': MLP,
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    model = models[name](**kwargs)

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
