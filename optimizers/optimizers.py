import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses.loss_functions import (
    MSELoss,
    MAELoss,
    BCELoss,
    CrossEntropyLoss,
    SparseCategoricalCrossEntropy,
)
from layers.sequence import Sequence
import layers.activations
from layers.feedforward import Linear


class Optimization:
    """
    Optimization class for training a model using SGD.
    """

    def __init__(
        self,
        model,
        X,
        y,
        epochs,
        loss_function,
        learning_rate,
        learning_rate_type="constant",
    ):
        self.model = model
        self.X = X
        self.y = y
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_type = learning_rate_type

    import torch

    def SGD(self):
        """
        Apply SGD (with optional momentum) to minimize the loss.

        Returns:
            list: Final model parameters.
        """
        momentum = 0.9

        # Fix params list once so velocities line up with params
        params = list(self.model.parameters())
        momentum_velocities = [torch.zeros_like(p) for p in params]

        for i in range(self.epochs):
            preds = self.model(self.X)
            loss = self.loss_function(preds, self.y)

            grads = torch.autograd.grad(loss, params)

            if self.learning_rate_type == "constant":
                for p, g in zip(params, grads):
                    if g is None:
                        continue
                    p.data -= self.learning_rate * g
            elif self.learning_rate_type == "momentum":
                for idx, (p, g) in enumerate(zip(params, grads)):
                    if g is None:
                        continue
                    momentum_velocities[idx] = (
                        momentum * momentum_velocities[idx] - self.learning_rate * g
                    )
                    p.data += momentum_velocities[idx]
            else:
                raise ValueError(
                    f"Unknown learning_rate_type: {self.learning_rate_type}"
                )

            print(f"Epoch {i+1}: Loss = {loss.item():.4f}")

        return params


if __name__ == "__main__":
    # Dummy data
    X = torch.randn((10, 3))  # 10 samples, 3 features
    y = torch.randn((10, 1))  # 10 targets

    # Define model
    model = Sequence(Linear(3, 4), Linear(4, 1))

    # Train
    trainer = Optimization(
        model=model,
        X=X,
        y=y,
        epochs=10,
        loss_function=MSELoss,
        learning_rate=0.01,
        learning_rate_type="momentum",
    )

    final_params = trainer.SGD()

    # Show final weights
    print("\nFinal Parameters:")
    for param in final_params:
        print(param)
