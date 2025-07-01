# Activation Output
A Python utility library for computing activation outputs in machine learning classification tasks. This package provides a unified interface for handling binary, multiclass, and multilabel classification scenarios.

## Features

- **Multiple Classification Types**: Support for binary, multiclass, and multilabel classification
- **Numerically Stable**: Implements stable computation methods to prevent overflow/underflow
- **Type Safe**: Full type hints and validation for robust usage
- **NumPy Integration**: Built on NumPy for efficient array operations

## Installation

This project uses `uv` for dependency management. To install:

```bash
# Clone the repository
git clone <repository-url>
cd krish-sandbox

# Install dependencies using uv
uv sync
```

## Requirements

- Python >= 3.13
- NumPy >= 2.3.1

## Usage

### Basic Usage

```python
import numpy as np
from krish_sandbox import activation_output

# Binary classification
logits = np.array([2.5, -1.0, 0.8])
probabilities = activation_output(logits, task_type='binary')
print(probabilities)  # [0.92414182 0.26894142 0.68997448]

# Multiclass classification
logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]])
probabilities = activation_output(logits, task_type='multiclass')
print(probabilities)
# [[0.24472847 0.66524096 0.09003057]
#  [0.30060961 0.33222499 0.3671654 ]]

# Multilabel classification
logits = np.array([[1.0, -0.5, 2.0], [0.1, 0.8, -0.3]])
probabilities = activation_output(logits, task_type='multilabel')
print(probabilities)
# [[0.73105858 0.37754067 0.88079708]
#  [0.52497919 0.68997448 0.42555748]]
```

### Task Types

#### Binary Classification
- **Input**: Scalar or 1D array of logits
- **Output**: Probabilities between 0 and 1
- **Activation**: Sigmoid function

#### Multiclass Classification
- **Input**: 2D array of logits (batch_size, num_classes)
- **Output**: Probability distribution over classes (sums to 1)
- **Activation**: Softmax function with numerical stability

#### Multilabel Classification
- **Input**: 2D array of logits (batch_size, num_labels)
- **Output**: Independent probabilities for each label
- **Activation**: Sigmoid function applied to each logit

## API Reference

### `activation_output(logits, task_type='binary')`

Computes activation output for classification tasks.

**Parameters:**
- `logits` (Union[npt.NDArray, Sequence]): Raw model outputs
- `task_type` (str): One of ['binary', 'multiclass', 'multilabel']. Defaults to 'binary'

**Returns:**
- `npt.NDArray`: Probabilities after applying appropriate activation function

**Raises:**
- `ValueError`: If logits array is empty or contains NaN/infinite values
- `ValueError`: If task_type is not one of the valid options

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```
