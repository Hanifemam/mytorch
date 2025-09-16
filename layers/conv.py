import torch
import torch.nn as nn


class Conv2DManual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        # [out_c, in_c, kh, kw]
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.bias)

        self.kh, self.kw = kh, kw

    def forward(self, x):  # x: [N, C_in, H, W]
        N, Cin, H, W = x.shape
        kh, kw = self.kh, self.kw
        Cout = self.weight.shape[0]

        Hout = H - kh + 1
        Wout = W - kw + 1
        assert Hout > 0 and Wout > 0, "Input too small for VALID conv with this kernel"

        y_batch = []
        # Pre-flatten weights once: [Cout, Cin*kh*kw]
        W_flat = self.weight.view(Cout, -1)

        for n in range(N):
            rows = []
            for i in range(Hout):
                cols = []
                for j in range(Wout):
                    # patch: [Cin, kh, kw]
                    patch = x[n, :, i : i + kh, j : j + kw]
                    v = torch.mv(W_flat, patch.reshape(-1)) + self.bias  # [Cout]
                    cols.append(v)  # list of [Cout]
                cols = torch.stack(cols, dim=0)  # [Wout, Cout]
                rows.append(cols)
            rows = torch.stack(rows, dim=0)  # [Hout, Wout, Cout]
            rows = rows.permute(2, 0, 1).contiguous()  # [Cout, Hout, Wout]
            y_batch.append(rows)

        return torch.stack(y_batch, dim=0)  # [N, Cout, Hout, Wout]


torch.manual_seed(0)
x = torch.randn(1, 3, 5, 5)  # [N, C, H, W]

manual = Conv2DManual(3, 2, 3)
builtin = nn.Conv2d(3, 2, 3, bias=True)

# Copy weights/bias so they match
with torch.no_grad():
    builtin.weight.copy_(manual.weight)
    builtin.bias.copy_(manual.bias)

y_manual = manual(x)
y_builtin = builtin(x)

print("Manual:\n", y_manual)
print("Builtin:\n", y_builtin)
print("Difference:", torch.abs(y_manual - y_builtin).max().item())
