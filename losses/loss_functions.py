"""This module contains custom loss functions for training neural networks.
loss functions for neural networks."""

import torch


def MSELoss(
    predictions,
    targets,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values from the model.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Computed MSE loss.
    """
    l2 = 0.0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return torch.mean(torch.square(predictions - targets)) + torch.as_tensor(
        l2, dtype=predictions.dtype, device=predictions.device
    )


def MAELoss(
    predictions,
    targets,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Computes the Mean Absolute Error (MAE) loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values from the model.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Computed MAE loss.
    """
    l2 = 0.0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return torch.mean(torch.abs(predictions - targets)) + torch.as_tensor(
        l2, dtype=predictions.dtype, device=predictions.device
    )


def BCELoss(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
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
    l2 = 0.0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return torch.mean(
        -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    ) + torch.as_tensor(l2, dtype=predictions.dtype, device=predictions.device)


def CrossEntropyLoss(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
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
    l2 = 0.0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return torch.mean(
        torch.sum(-targets * torch.log(predictions), axis=1)
    ) + torch.as_tensor(l2, dtype=predictions.dtype, device=predictions.device)


def SparseCategoricalCrossEntropy(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
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
    l2 = 0.0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return torch.mean(loss) + torch.as_tensor(
        l2, dtype=predictions.dtype, device=predictions.device
    )


def l2_regularizer(parameters, decay=1e-3):
    """
    Computes an L2 penalty term over a set of parameters.

    Args:
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor):
            Parameters to regularize. Must be non-empty. For convenience, a single
            tensor/variable is also accepted.
        decay (float, optional): L2 coefficient λ. The returned value is
            `λ * Σ_i reduce_sum(square(p_i))`. Defaults to 1e-3.

    Returns:
        tf.Tensor: Scalar tensor representing the L2 penalty.

    Notes:
        This helper does not guard against an empty list; callers should pass a non-empty
        sequence or check before calling (as done in the loss functions above).
    """
    if parameters is None or decay == 0.0:
        return 0.0
    if isinstance(parameters, (torch.Tensor, torch.Tensor)):
        params = [parameters]
    else:
        params = list(parameters)
    if len(params) == 0:
        return 0.0
    return decay * sum(p.pow(2).sum() for p in params)


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

if __name__ == "__main__":
    import torch

    # Tiny smoke test: compare MSE with/without L2
    W = torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float32, requires_grad=True)
    params = [W]
    decay = 1e-2

    preds = torch.tensor([[0.5], [0.2], [0.9]], dtype=torch.float32)
    targs = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    # No regularizer
    no_reg = ((preds - targs) ** 2).mean()

    # With L2 regularizer (sum of squares across params)
    l2 = decay * sum(p.pow(2).sum() for p in params)
    with_l2 = no_reg + l2

    print(f"MSE (no reg): {float(no_reg):.6f}")
    print(f"MSE (+L2):    {float(with_l2):.6f}")
