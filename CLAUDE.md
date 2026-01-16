# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating PAC learning theory - generalization bounds, VC dimension, growth functions, and the relationship between hypothesis complexity and learnability. Uses MNIST as a testbed for experiments. Designed for rapid prototyping on Google Colab (A100 recommended).

## Project Structure

```
MNIST_AI/
├── utils/              # Reusable utilities (import into notebooks)
│   ├── data.py         # Dataset loading, clustering, subsetting
│   ├── models.py       # Model architectures (SmallCNN, TinyCNN, MLP)
│   ├── training.py     # Training loops (single and parallel)
│   └── evaluation.py   # Accuracy, cross-evaluation, statistics
├── notebooks/          # Experiment notebooks
└── results/            # Saved outputs (git-tracked)
```

## Colab Workflow

```python
# Clone and import
!git clone https://github.com/Caleb-Briggs/MNIST_AI.git
%cd MNIST_AI

from utils.data import load_mnist, create_clusters, get_device
from utils.models import SmallCNN, create_model
from utils.training import train_model, train_parallel
from utils.evaluation import evaluate, cross_evaluate, compute_stats
```

## Key Utilities

### Data (`utils/data.py`)
- `load_mnist(device, train=True)` - Returns (images, labels) tensors on device
- `create_clusters(num_samples, sizes, seed)` - Create disjoint clusters
- `create_digit_subset(images, labels, digits)` - Filter to specific digits

### Models (`utils/models.py`)
- `SmallCNN(num_classes=10)` - ~100k params, 3 conv layers
- `TinyCNN(num_classes=10)` - ~10k params, 2 conv layers
- `MLP(hidden_sizes, num_classes)` - Configurable fully-connected
- `create_model(name, device, **kwargs)` - Factory function

### Training (`utils/training.py`)
- `train_model(model, images, labels, indices, epochs, lr, ...)` - Single model
- `train_parallel(models, images, labels, list_of_indices, ...)` - Multiple models

### Evaluation (`utils/evaluation.py`)
- `evaluate(model, images, labels, indices)` - Single accuracy
- `cross_evaluate(models, images, labels, clusters, train_indices)` - MxM matrix
- `compute_stats(matrix)` - Mean, std, percentiles from accuracy matrix

## Design Principles

1. **Notebooks are primary** - All experiments live in notebooks; utils are thin helpers
2. **Minimal abstraction** - Only abstract truly reusable code
3. **Git-versioned results** - Commit outputs to `results/` for reproducibility
