"""This module contains custom loss functions for training neural networks.
loss functions for neural networks."""

import torch


def MSELoss(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values from the model.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Computed MSE loss.
    """
    return torch.mean(torch.square(predictions - targets))


def MAELoss(predictions, targets):
    """
    Computes the Mean Absolute Error (MAE) loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values from the model.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Computed MAE loss.
    """
    return torch.mean(torch.abs(predictions - targets))


def BCELoss(predictions, targets, epsilon=1e-7):
    """
    Binary Cross Entropy loss.

    Args:
        predictions (torch.Tensor): Predicted probabilities (after sigmoid).
        targets (torch.Tensor): Ground truth labels (0 or 1).
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar BCE loss.
    """
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)  # avoid log(0)
    return torch.mean(
        -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    )


def CrossEntropyLoss(predictions, targets, epsilon=1e-7):
    """
    Cross Entropy loss for multi-class classification.

    Args:
        predictions (torch.Tensor): Predicted probabilities (after softmax).
        targets (torch.Tensor): One-hot encoded ground truth labels.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar cross entropy loss.
    """
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    return torch.mean(torch.sum(-targets * torch.log(predictions), axis=1))


def SparseCategoricalCrossEntropy(predictions, targets, epsilon=1e-7):
    """
    Computes the Sparse Categorical Cross Entropy loss.

    Args:
        predictions (torch.Tensor): Predicted probabilities, shape (batch_size, num_classes).
        targets (torch.Tensor): Integer class labels, shape (batch_size,).
        epsilon (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    predictions = torch.clamp(predictions, min=epsilon, max=1 - epsilon)
    targets = targets.view(-1)  # flatten if needed
    batch_indices = torch.arange(targets.shape[0])
    indices = torch.stack([batch_indices, targets], dim=1)
    true_probs = predictions[indices[:, 0], indices[:, 1]]
    loss = -torch.log(true_probs)
    return torch.mean(loss)


# === 1. MSELoss (regression)
print("=== MSELoss ===")
preds = torch.tensor([[0.5], [0.2], [0.9]], dtype=torch.float32)
targets = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
print("MSE Loss:", MSELoss(preds, targets).item())

# === 2. MAELoss (regression)
print("\n=== MAELoss ===")
print("MAE Loss:", MAELoss(preds, targets).item())

# === 3. BCELoss (binary classification)
print("\n=== BCELoss ===")
preds_bce = torch.tensor([[0.9], [0.2], [0.6]], dtype=torch.float32)
targets_bce = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
print("BCE Loss:", BCELoss(preds_bce, targets_bce).item())

# === 4. CrossEntropyLoss (multi-class, one-hot labels)
print("\n=== CrossEntropyLoss ===")
preds_ce = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=torch.float32)
targets_ce = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
print("Cross Entropy Loss:", CrossEntropyLoss(preds_ce, targets_ce).item())

# === 5. SparseCategoricalCrossEntropy (multi-class, integer labels)
print("\n=== SparseCategoricalCrossEntropy ===")
targets_sparse = torch.tensor(
    [1, 0], dtype=torch.int64
)  # same as above, but not one-hot
print(
    "Sparse Cross Entropy Loss:",
    SparseCategoricalCrossEntropy(preds_ce, targets_sparse).item(),
)
