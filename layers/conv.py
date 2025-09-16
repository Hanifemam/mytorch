import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DManual(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super().__init__()
        # kernel size
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.kh, self.kw = kh, kw

        # stride (h, w)
        if isinstance(stride, int):
            self.sh = self.sw = stride
        else:
            self.sh, self.sw = stride

        # padding (h, w)
        if isinstance(padding, int):
            self.ph = self.pw = padding
        else:
            self.ph, self.pw = padding

        # weights: [Cout, Cin, kh, kw] in PyTorch
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):  # x: [N, Cin, H, W]
        # pad first (symmetric on H and W)
        if self.ph or self.pw:
            x = F.pad(
                x, pad=(self.pw, self.pw, self.ph, self.ph), mode="constant", value=0
            )

        N, Cin, Hp, Wp = x.shape  # sizes AFTER padding
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        Cout = self.weight.shape[0]

        # output sizes (VALID conv on the padded input)
        Hout = (Hp - kh) // sh + 1
        Wout = (Wp - kw) // sw + 1
        assert Hout > 0 and Wout > 0, "Input too small for given kernel/stride/padding."

        # flatten weights once: [Cout, Cin*kh*kw]
        W_flat = self.weight.view(Cout, -1)

        # allocate output
        y = x.new_empty((N, Cout, Hout, Wout))

        for n in range(N):
            for i_out in range(Hout):
                i0 = i_out * sh
                for j_out in range(Wout):
                    j0 = j_out * sw
                    patch = x[n, :, i0 : i0 + kh, j0 : j0 + kw]  # [Cin, kh, kw]
                    y[n, :, i_out, j_out] = W_flat @ patch.reshape(-1) + (
                        self.bias if self.bias is not None else 0.0
                    )

        return y


torch.manual_seed(0)

# toy input
x = torch.randn(1, 3, 8, 7)  # [N, C, H, W]

# choose any padding/stride to test
p = 1
s = (2, 3)

manual = Conv2DManual(3, 4, kernel_size=3, padding=p, stride=s, bias=True)
builtin = nn.Conv2d(3, 4, kernel_size=3, padding=p, stride=s, bias=True)

# make both layers use identical weights/bias
with torch.no_grad():
    builtin.weight.copy_(manual.weight)
    builtin.bias.copy_(manual.bias)

y_manual = manual(x)
y_builtin = builtin(x)

print("shapes:", y_manual.shape, y_builtin.shape)
print("max |diff|:", (y_manual - y_builtin).abs().max().item())
