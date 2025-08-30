import torch
import torch.nn as nn
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class GatedFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.w_gate = nn.Linear(dim, hidden_dim)
        self.w_in = nn.Linear(dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, dim)
        self.silu = nn.SiLU()
    def forward(self, x):
        h_gate = self.silu(self.w_gate(x))
        h_act = self.silu(self.w_in(x))
        return self.w_out(h_act * h_gate)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, is_cross_attention=False):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.is_cross_attention = is_cross_attention
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False) if is_cross_attention else None
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False) if not is_cross_attention else None
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x, context=None, mask=None, pos_emb=None):
        h = self.heads
        q = self.to_q(x)
        k, v = (self.to_kv(context) if self.is_cross_attention else self.to_qkv(x)).chunk(2 if self.is_cross_attention else 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        if pos_emb is not None:
            q, k = apply_rotary_pos_emb(pos_emb, q), apply_rotary_pos_emb(pos_emb, k)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), (nn.LayerNorm(dim) if is_cross_attention else None)
        self.attn = Attention(dim, heads, is_cross_attention)
        self.norm_ffn, self.ffn = nn.LayerNorm(dim), GatedFFN(dim)
    def forward(self, x, context=None, mask=None, pos_emb=None):
        context = self.norm_kv(context) if self.is_cross_attention else None
        x = self.attn(self.norm_q(x), context=context, mask=mask, pos_emb=pos_emb) + x
        return self.ffn(self.norm_ffn(x)) + x

class CrissCrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        h_grid = w_grid = int(math.sqrt(n))
        padding = h_grid * w_grid - n
        if padding > 0: x = torch.cat([x, torch.zeros(b, padding, x.shape[2], device=x.device)], dim=1)
        q, k, v = map(lambda t: rearrange(t(x), 'b (h w) (heads d) -> b heads h w d', heads=h, h=h_grid), (self.to_q, self.to_k, self.to_v))
        dots_row = torch.einsum('b h i j d, b h i k d -> b h i j k', q, k) * self.scale
        out_row = torch.einsum('b h i j k, b h i k d -> b h i j d', dots_row.softmax(dim=-1), v)
        q_col, k_col, v_col = map(lambda t: t.transpose(-2, -3), (q, k, v))
        dots_col = torch.einsum('b h j i d, b h k i d -> b h j i k', q_col, k_col) * self.scale
        out_col = torch.einsum('b h j i k, b h k i d -> b h j i d', dots_col.softmax(dim=-1), v_col)
        out = rearrange(out_row + out_col.transpose(-2, -3), 'b heads h w d -> b (h w) (heads d)')
        return self.to_out(out[:, :-padding] if padding > 0 else out)

class CrissCrossTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), CrissCrossAttention(dim, heads)
        self.norm2, self.ffn = nn.LayerNorm(dim), GatedFFN(dim)
    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.ffn(self.norm2(x)) + x

class VQTokenizer(nn.Module):
    def __init__(self, num_channels, d_model, codebook_size):
        super().__init__()
        self.stft_n_fft = 256
        self.vq_proj = nn.Linear(self.stft_n_fft // 2 + 1, d_model)
        self.vq_vae = VectorQuantize(dim=d_model, codebook_size=codebook_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_channels, d_model))
    def forward(self, x):
        quantized, _, commit_loss = self.vq_vae(self.vq_proj(x))
        return quantized + self.pos_embedding, None, commit_loss

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.patch_size, self.proj = patch_size, nn.Linear(patch_size, embedding_dim)
    def forward(self, x):
        return self.proj(x.unfold(2, self.patch_size, self.patch_size))

class MultiScaleConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels, n_convs, kernel_sizes, strides, out_channels):
        super().__init__()
        convs = []
        for i in range(n_convs):
            convs.append(nn.Sequential(
                nn.Conv1d(in_channels if i==0 else out_channels, out_channels, kernel_sizes[i], strides[i], padding=kernel_sizes[i]//2),
                nn.GELU(), nn.LayerNorm([out_channels, -1]) # 채널, 시간 축에 대한 정규화
            ))
        self.convs = nn.ModuleList(convs)
    def forward(self, x):
        for conv in self.convs: x = conv(x)
        return x.mean(dim=-1) # 시간 축에 대해 평균 풀링

class PSDFeatureExtractor(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
    def forward(self, x):
        stft = torch.stft(x, self.n_fft, self.hop_length, return_complex=True, window=torch.hann_window(self.n_fft, device=x.device))
        psd = torch.abs(stft)**2
        return psd.mean(dim=-1)
