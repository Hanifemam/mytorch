from .activations import ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
from layers.feedforward import Linear


class Sequence:
    """
    A class to represent a sequence of layers in a neural network.

    This class allows for the sequential application of multiple layers,
    enabling the construction of complex neural network architectures.
    """

    def __init__(self, *layers):
        """
        Initializes the Sequence with a list of layers.

        Args:
            *layers: Variable number of layer instances to be included in the sequence.
        """
        self.layers = layers

    def __call__(self, x):
        """
        Applies the sequence of layers to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be processed by the sequence.

        Returns:
            torch.Tensor: Output tensor after applying all layers in sequence.
        """
        for layer in self.layers:
            if not callable(layer):
                raise TypeError(f"Layer {layer} must be callable")
            else:
                x = layer(x)
        print(x)
        return x

    def parameters(self):
        """
        Collects the parameters from all layers in the sequence.

        Returns:
            list: A flat list of all trainable parameters from all layers.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters)
        return params

    def __repr__(self):
        """
        Returns a string representation of the sequence of layers.

        Returns:
            str: Human-readable list of layers in sequence.
        """
        layer_strs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequence(\n" + "\n".join(layer_strs) + "\n)"


print("TEST")
model = Sequence(Linear(3, 4), ReLU(), Linear(4, 1), Sigmoid())
