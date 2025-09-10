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

    def __init__(self, input_size, output_size, residual=False):
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
        self.residual = residual

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
        if self.residual:
            return torch.matmul(x, self._W) + self._b + x
        else:
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


class BatchNormalization:
    """
    Batch Normalization for 2D inputs shaped (N, F).
    Keeps your original API: _W (gamma), _b (beta), parameters, W/b properties.
    """

    def __init__(self, input_size, epsilon=1e-5, momentum=0.99):
        self.input_size = int(input_size)
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)

        self._W = torch.tensor(
            torch.ones((self.input_size,), dtype=torch.float32),  # gamma
            requires_grad=True,
        )
        self._b = torch.tensor(
            torch.zeros((self.input_size,), dtype=torch.float32),  # beta
            requires_grad=True,
        )

        self.moving_mean = torch.tensor(
            torch.zeros((self.input_size,), dtype=torch.float32),
            requires_grad=False,
        )
        self.moving_var = torch.tensor(
            torch.ones((self.input_size,), dtype=torch.float32),
            requires_grad=False,
        )

    def __call__(self, x, training=True):
        """Apply BatchNorm.
        Args:
            x: (N, F) tensor
            training (bool): if True use batch stats and update running stats,
            else use moving stats.
        """
        x = torch.tensor(x, dtype=torch.float32)

        if training:
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)

            self.moving_mean = torch.tensor(
                self.momentum * self.moving_mean + (1.0 - self.momentum) * batch_mean
            )

            self.moving_var = torch.tensor(
                self.momentum * self.moving_var + (1.0 - self.momentum) * batch_var
            )

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.moving_mean, self.moving_var

        # Normalize with standard deviation (sqrt(variance)), not variance
        x_hat = (x - mean) / torch.sqrt(var + self.epsilon)  # (N, F)

        # Affine transform: gamma (W) and beta (b)
        return x_hat * self._W + self._b

    # ---- Compatibility with your original API ----
    @property
    def parameters(self):
        return [self._W, self._b]

    @property
    def W(self):  # gamma
        return self._W

    @W.setter
    def W(self, value):
        value = torch.tensor(value, dtype=torch.float32)
        if value.shape != self._W.size():
            raise ValueError(f"W shape {value.shape} != {self._W.size()}")
        self._W = value

    @property
    def b(self):  # beta
        return self._b

    @b.setter
    def b(self, value):
        value = torch.tensor(value, dtype=torch.float32)
        if value.shape != self._b.size:
            raise ValueError(f"b shape {value.shape} != {self._b.shape}")
        self._b = value


class LayerNormalization:
    """
    Batch Normalization for 2D inputs shaped (N, F).
    Keeps your original API: _W (gamma), _b (beta), parameters, W/b properties.
    """

    def __init__(self, input_size, epsilon=1e-5):
        self.input_size = int(input_size)
        self.epsilon = float(epsilon)

        self._W = torch.tensor(
            torch.ones((self.input_size,), dtype=torch.float32),  # gamma
            requires_grad=True,
        )
        self._b = torch.tensor(
            torch.zeros((self.input_size,), dtype=torch.float32),  # beta
            requires_grad=True,
        )

    def __call__(self, x):
        """Apply LayerNorm.
        Args:
            x: (N, F) tensor
        returns:
            (N, F) tensor after layer normalization.
        """
        x = torch.tensor(x, dtype=torch.float32)

        batch_mean = torch.mean(x, axis=1, keepdim=True)
        batch_var = torch.var(x, axis=1, keepdim=True)

        mean, var = batch_mean, batch_var

        # Normalize with standard deviation (sqrt(variance)), not variance
        x_hat = (x - mean) / torch.sqrt(var + self.epsilon)  # (N, F)

        # Affine transform: gamma (W) and beta (b)
        return x_hat * self._W + self._b

    # ---- Compatibility with your original API ----
    @property
    def parameters(self):
        return [self._W, self._b]

    @property
    def W(self):  # gamma
        return self._W

    @W.setter
    def W(self, value):
        value = torch.tensor(value, dtype=torch.float32)
        if value.shape != self._W.size():
            raise ValueError(f"W shape {value.shape} != {self._W.size()}")
        self._W = value

    @property
    def b(self):  # beta
        return self._b

    @b.setter
    def b(self, value):
        value = torch.tensor(value, dtype=torch.float32)
        if value.shape != self._b.size:
            raise ValueError(f"b shape {value.shape} != {self._b.shape}")
        self._b = value


# Test the Linear layer implementation
if __name__ == "__main__":
    # layer = Linear(input_size=3, output_size=2)

    # # Create dummy input: batch of 4 samples, each with 3 features
    # x = torch.randn((4, 3))

    # # Forward pass
    # output = layer(x)

    # print("Input:\n", x)
    # print("Output:\n", output)
    # print("Weights:\n", layer.W)
    # print("Weights shape:", layer.W.shape)
    # print("Biases:\n", layer.b)
    layer = LayerNormalization(input_size=3)

    # Random input: 4 samples, 3 features
    x = torch.normal(mean=5.0, std=2.0, size=(4, 3))

    # Forward pass (training mode)
    output = layer(x, training=True)

    print("Input:\n", x)
    print("Output (normalized):\n", output)
    print("Gamma (W):\n", layer.W)
    print("Gamma shape:", layer.W.shape)
    print("Beta (b):\n", layer.b)
    print("Beta shape:", layer.b.shape)
