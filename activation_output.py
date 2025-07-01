import numpy as np
import numpy.typing as npt
from typing import Union
from collections.abc import Sequence, Callable

def activation_output(logits: Union[npt.NDArray, Sequence], task_type: str = 'binary') -> npt.NDArray:
    """
    Computes activation output for classification tasks.
    
    Args:
    - logits: Raw model outputs
    - task_type: One of ['binary', 'multiclass', 'multilabel']
    
    Returns:
    - npt.NDArray: Probabilities after applying appropriate activation
    """
    logits: npt.NDArray = np.array(logits)
    
    # Input validation
    if not logits.size:
        raise ValueError("Empty logits array provided")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinite values")
    
    valid_types: Sequence[str] = ['binary', 'multiclass', 'multilabel']
    if task_type not in valid_types:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be one of {valid_types}")

    sigmoid: Callable[[npt.NDArray], npt.NDArray] = lambda x : 0.5 * (1 + np.tanh(x * 0.5))

    if task_type == 'binary':
        """
        Input shape: Scalar or (batch_size,)
        Directly apply sigmoid function to the logits
        Output shape: Scalar or (batch_size,)
        """
        return sigmoid(logits)

    elif task_type == 'multiclass':
        """
        Input shape: (batch_size, num_classes) or (num_classes,)
        Subtract the maximum logit from each to prevent overflow
        Apply softmax function to the logits
        Output shape: (batch_size, num_classes) or (num_classes,)
        """
        logits_stable: npt.NDArray = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits: npt.NDArray = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    elif task_type == 'multilabel':
        """
        Input shape: (batch_size, num_labels) or (num_labels,)
        Apply sigmoid function to each logits
        Output shape: (batch_size, num_labels) or (num_labels,)
        """
        return sigmoid(logits)