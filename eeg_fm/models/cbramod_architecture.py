import torch
import torch.nn as nn
from einops import rearrange
from models.components import CrissCrossTransformerBlock, PatchEmbedding

class CBraMod(nn.Module):
    def __init__(self, num_classes, num_channels, embedding_dim, depth, heads, finetune_head_mode='cls_token', **kwargs):
        super().__init__()
        patch_size = 32
        self.finetune_head_mode = finetune_head_mode
        self.num_channels = num_channels
        
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embedding_dim=embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_channels + 1, embedding_dim))
        self.encoder = nn.ModuleList([CrissCrossTransformerBlock(embedding_dim, heads) for _ in range(depth)])
        
        n_class = next(iter(num_classes.values())) if isinstance(num_classes, dict) else num_classes
        self.mlp_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, n_class))
        self.reconstruction_head = nn.Linear(embedding_dim, patch_size)

    def forward(self, x, mode='finetune', **kwargs):
        b, t, c, fs = x.shape
        if c != self.num_channels:
             x = x[:, :, :self.num_channels, :]
        
        x_reshaped = rearrange(x, 'b t c fs -> b c (t fs)')
        patches = self.patch_embed(x_reshaped)
        x_pooled = torch.mean(patches, dim=2)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x_pooled), dim=1)
        x_with_cls += self.pos_embedding
        
        encoded_features = x_with_cls
        for block in self.encoder:
            encoded_features = block(encoded_features)
            
        if mode == 'pretrain_ssl':
            return self.reconstruction_head(encoded_features[:, 1:, :])
        else:
            if self.finetune_head_mode == 'cls_token':
                features = encoded_features[:, 0]
            elif self.finetune_head_mode == 'mean_pool':
                features = encoded_features[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(f"Unknown mode: {self.finetune_head_mode}")
            return self.mlp_head(features)
