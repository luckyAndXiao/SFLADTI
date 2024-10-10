import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

    def forward(self, x, y=None):
        B, N, C = x.shape
        if y is None:
            y = x

        q = self.q(y)
        kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        num = int(q.shape[1] ** 0.5)
        #v -> q
        feature_map = rearrange(q, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SiameseFLA(nn.Module):
    def __init__(self, dim, heads=8):
        super(SiameseFLA, self).__init__()
        self.flatt = FocusedLinearAttention(dim=dim, num_heads=heads)
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        # self.swish = lambda x: x * torch.sigmoid(x)
        self.gelu = nn.GELU()
    def forward(self, x, y, z):

        xz = self.flatt(z, x)
        yz = self.flatt(z, y)

        # xz = xz * self.swish(x)
        # yz = yz * self.swish(y)

        # xz = xz + x
        # yz = yz + y

        xz = xz * self.gelu(x)
        yz = yz * self.gelu(y)

        v1 = reduce(xz, "B H W -> B W", "max")
        v2 = reduce(yz, "B H W -> B W", "max")

        f = self.tanh(torch.cat((v1, v2), dim=-1))

        return f



if __name__ == "__main__":
    x = torch.randn((64, 324, 128))
    y = torch.randn((64, 324, 128))
    z = torch.randn((64, 1225, 128))

    model = SiameseFLA(128)

    res = model(x, y, z)

    print(res.shape)



