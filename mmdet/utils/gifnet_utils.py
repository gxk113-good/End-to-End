import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out

class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std  = x.std((2, 3), keepdim=True, unbiased=False) + self.eps
        x_norm = (x - mean) / std
        rescale = self.meta1(std)
        rebias  = self.meta2(mean)
        out = x_norm * self.weight + self.bias
        return out, rescale, rebias

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

def get_relative_positions(window_size):
    coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))
    coords_flat = coords.flatten(1)
    rel_pos = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel_pos = rel_pos.permute(1, 2, 0).contiguous()
    return torch.sign(rel_pos) * torch.log(1. + rel_pos.abs())

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.register_buffer("relative_positions", get_relative_positions(window_size))
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        B_, N, C = q.shape
        head_dim = C // self.num_heads
        
        q = q.view(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        rel_bias = self.meta(self.relative_positions)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn += rel_bias
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=98, embed_dim=98, kernel_size=3):
        super().__init__()
        padding = (kernel_size - patch_size + 1) // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, 
                              stride=patch_size, padding=padding, padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=1, out_chans=98, embed_dim=98, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=padding, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, out_features or in_features, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = RLN(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = RLN(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.window_size = window_size

    def check_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        identity = x
        
        # 窗口注意力
        x, rescale, rebias = self.norm1(x)
        B, C, H, W = x.shape
        x = self.check_size(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 窗口划分
        x = window_partition(x, self.window_size)  # [num_windows*B, window_size*window_size, C]
        
        # 注意力计算
        x = self.attn(x, x, x)
        
        # 窗口合并
        x = window_reverse(x, self.window_size, H, W)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 恢复原始尺寸
        x = x[:, :, :identity.shape[2], :identity.shape[3]]
        x = x * rescale + rebias
        x = identity + x
        
        # MLP
        identity = x
        x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        x = x * rescale + rebias
        x = identity + x
        
        return x

class TransformerNet(nn.Module):
    def __init__(self, in_chans=98, embed_dim=98, depth=2, num_heads=2, window_size=8):
        super().__init__()
        self.patch_embed = PatchEmbed(1, in_chans, embed_dim, 3)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, window_size)
            for _ in range(depth)
        ])
        self.patch_unembed = PatchUnEmbed(1, embed_dim, embed_dim, 3)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.patch_unembed(x)
        return x

class SharedFeatureExtractor(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans*2, 32, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(32+in_chans*2, 32, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(32+32+in_chans*2, 32, 3, padding=1, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        x1 = self.relu(self.conv1(x0))
        x2 = self.relu(self.conv2(torch.cat([x0, x1], dim=1)))
        x3 = self.relu(self.conv3(torch.cat([x0, x1, x2], dim=1)))
        return torch.cat([x0, x1, x2, x3], dim=1)
