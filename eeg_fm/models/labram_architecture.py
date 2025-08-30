import torch
import torch.nn as nn
from models.components import VQTokenizer

class LaBraM(nn.Module):
    def __init__(self, num_classes, num_channels, d_model, nhead, num_encoder_layers, dim_feedforward, codebook_size, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = VQTokenizer(num_channels, d_model, codebook_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        n_class = next(iter(num_classes.values())) if isinstance(num_classes, dict) else num_classes
        self.finetune_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_class))
        self.reconstruction_head = nn.Linear(d_model, self.tokenizer.stft_n_fft // 2 + 1)
        
    def forward(self, x, mode='finetune', mask=None, return_features_only=False, **kwargs):
        quantized, _, commit_loss = self.tokenizer(x)
        if mask is not None:
            quantized = quantized * mask.unsqueeze(-1)
        encoded = self.transformer_encoder(quantized)
        
        if mode == 'pretrain':
            return self.reconstruction_head(encoded), x, commit_loss
        else:
            features = encoded.mean(dim=1)
            if return_features_only:
                return features
            return self.finetune_head(features)
