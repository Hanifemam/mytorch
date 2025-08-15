import torch
import numpy as np


class Normalization:
    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(torch.float32)
        mean = torch.mean(x, axis=0, keepdims=True)
        std = torch.std(x, axis=0, keepdims=True)
        return (x - mean) / (std + self.epsilon)


if __name__ == "__main__":
    import numpy as np

    data = np.random.randn(5, 3) * 5 + 20  # 5 samples, 3 features
    print("Unnormalized data: ", data)
    norm_layer = Normalization()
    normed = norm_layer(data)
    print("Normalized data: ", normed)
