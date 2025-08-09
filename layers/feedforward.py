import torch
import numpy as np


class Linear:
    """
    A custom implementation of a fully connected (dense) linear layer using PyTorch.

    This layer performs the operation: output = activation(x @ W + b),
    where `x` is the input tensor, `W` is the weight matrix,
    and `b` is the bias vector. An activation function can be applied.

    Attributes:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        _W (torch.Tensor): The weight matrix of shape (input_size, output_size).
        _b (torch.Tensor): The bias vector of shape (output_size,).
        activation (str): The name of the activation function to apply.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the Linear layer with random weights and biases.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            activation (str): Name of the activation function (default: 'ReLU').
        """
        self.input_size = input_size
        self.output_size = output_size
        mean = 0.0
        std = np.sqrt(2.0 / (input_size))
        self._W = torch.normal(
            size=(input_size, output_size),
            mean=mean,
            std=std,
            dtype=torch.float32,
            requires_grad=True,
        )
        self._b = torch.normal(
            size=(output_size,),
            mean=mean,
            std=std,
            dtype=torch.float32,
            requires_grad=True,
        )

    def __call__(self, x):
        """
        Enables the layer to be called like a function.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor.

        Returns:
            torch.Tensor: Output after applying the linear transformation.
        """
        return self.forward(x)

    def __repr__(self):
        """
        Returns a string representation of the Linear layer.

        Returns:
            str: A human-readable description of the layer.
        """
        return f"Linear(input_size={self.input_size}, output_size={self.output_size})"

    def forward(self, x):
        """
        Forward pass of the Linear layer.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return torch.matmul(x, self._W) + self._b

    @property
    def parameters(self):
        """
        Returns the trainable parameters of the layer.

        Returns:
            list: A list containing the weight matrix and bias vector.
        """
        return [self._W, self._b]

    @property
    def W(self):
        """
        Accesses the weight matrix.

        Returns:
            torch.Tensor: The weight matrix.
        """
        return self._W

    @W.setter
    def W(self, value):
        """
        Sets the weight matrix with a new value.

        Args:
            value (np.ndarray or torch.Tensor): New weight values.

        Raises:
            ValueError: If the input is not a valid array or tensor.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self._W = value

    @property
    def b(self):
        """
        Accesses the bias vector.

        Returns:
            torch.Tensor: The bias vector.
        """
        return self._b

    @b.setter
    def b(self, value):
        """
        Sets the bias vector with a new value.

        Args:
            value (np.ndarray or torch.Tensor): New bias values.

        Raises:
            ValueError: If the input is not a valid array or tensor.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self._b = value


# Test the Linear layer implementation
if __name__ == "__main__":
    layer = Linear(input_size=3, output_size=2)

    # Create dummy input: batch of 4 samples, each with 3 features
    x = torch.randn((4, 3))

    # Forward pass
    output = layer(x)

    print("Input:\n", x)
    print("Output:\n", output)
    print("Weights:\n", layer.W)
    print("Weights shape:", layer.W.shape)
    print("Biases:\n", layer.b)
