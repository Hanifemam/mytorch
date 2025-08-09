import torch
import sys
import os
import numpy as np

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

    def SGD(self):
        """
        Apply SGD (constant LR, momentum, or linear schedule) to minimize the loss.
        Returns:
            list: Final model parameters.
        """
        momentum = 0.9

        # Fix params once so velocity buffers align
        params = list(self.model.parameters())
        velocities = [torch.zeros_like(p) for p in params]

        # Precompute schedule if needed (per-epoch schedule here)
        if self.learning_rate_type == "linear":
            lr_schedule = self.linear_lr_schedule(total_steps=self.epochs)

        for i in range(self.epochs):
            preds = self.model(self.X)
            loss = self.loss_function(preds, self.y)
            grads = torch.autograd.grad(loss, params)

            with torch.no_grad():
                if self.learning_rate_type == "constant":
                    for p, g in zip(params, grads):
                        p -= self.learning_rate * g

                elif self.learning_rate_type == "momentum":
                    for idx, (p, g) in enumerate(zip(params, grads)):
                        velocities[idx] = (
                            momentum * velocities[idx] - self.learning_rate * g
                        )
                        p += velocities[idx]

                elif self.learning_rate_type == "linear":
                    lr_t = float(lr_schedule[i])
                    for p, g in zip(params, grads):
                        p -= lr_t * g
                else:
                    raise ValueError(
                        f"Unknown learning_rate_type: {self.learning_rate_type}"
                    )

            print(f"Epoch {i+1}: Loss = {loss.item():.4f}")

        return params

    def linear_lr_schedule(
        self, total_steps, base_lr=1e-3, end_frac=0.01, warmup_ratio=0.05
    ):
        # Robust guards
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(round(total_steps * warmup_ratio)), 0)
        min_lr = base_lr * end_frac
        decay_steps = max(total_steps - warmup_steps, 1)

        lrs = []
        for t in range(total_steps):
            if warmup_steps > 0 and t < warmup_steps:
                # Linear warmup 0 -> base_lr (use t+1 to avoid 0 when desired)
                lr = base_lr * ((t + 1) / warmup_steps)
            else:
                # Linear decay base_lr -> min_lr
                progress = (t - warmup_steps) / decay_steps
                progress = min(max(progress, 0.0), 1.0)
                lr = base_lr - progress * (base_lr - min_lr)
            lrs.append(lr)
        return np.array(lrs, dtype=np.float32)

    def power_law_lr_schedule():
        pass

    def exponential_lr_schedule():
        pass


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
