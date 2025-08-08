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

    def __init__(self, model, X, y, epochs, loss_function, learning_rate):
        self.model = model
        self.X = X
        self.y = y
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def SGD(self):
        """
        Apply Stochastic Gradient Descent to minimize the loss.

        Returns:
            list: The final model parameters (weights and biases).
        """
        for i in range(self.epochs):

            prediction = self.model(self.X)
            loss = self.loss_function(prediction, self.y)

            parameters = self.model.parameters()
            grads = torch.autograd.grad(loss, parameters, retain_graph=True)

            for param, grad in zip(parameters, grads):
                param.data -= self.learning_rate * grad

            print(f"Epoch {i+1}: Loss = {loss.item():.4f}")

        return self.model.parameters()


if __name__ == "__main__":
    # Dummy data
    X = torch.randn((10, 3))  # 10 samples, 3 features
    y = torch.randn((10, 1))  # 10 targets

    # Define model
    model = Sequence(Linear(3, 4), Linear(4, 1))

    # Train
    trainer = Optimization(
        model=model, X=X, y=y, epochs=10, loss_function=MSELoss, learning_rate=0.01
    )

    final_params = trainer.SGD()

    # Show final weights
    print("\nFinal Parameters:")
    for param in final_params:
        print(param)
