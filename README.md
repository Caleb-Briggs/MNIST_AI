# MNIST_AI

Research project investigating PAC learning theory using MNIST as a testbed.

## Research Focus

- Generalization bounds and sample complexity
- VC dimension and growth functions
- Relationship between hypothesis complexity and learnability
- Empirical validation of theoretical learning bounds

## Quick Start (Google Colab)

```python
!git clone https://github.com/Caleb-Briggs/MNIST_AI.git
%cd MNIST_AI

from utils.data import load_mnist, create_clusters, get_device
from utils.models import SmallCNN
from utils.training import train_model
from utils.evaluation import evaluate

# Load data
device = get_device()
images, labels = load_mnist(device)

# Train a model
model = SmallCNN().to(device)
train_model(model, images, labels, indices=range(1000), epochs=10)

# Evaluate
acc = evaluate(model, images, labels)
print(f"Accuracy: {acc:.4f}")
```

## Project Structure

```
├── utils/          # Reusable utilities
├── notebooks/      # Experiment notebooks
└── results/        # Saved outputs (git-tracked)
```

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- scipy
- tqdm

Google Colab with A100 GPU recommended for larger experiments.
