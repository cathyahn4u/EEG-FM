import torch
import torch.nn as nn
from models.components import (
    TransformerBlock, RotaryEmbedding, MultiScaleConvFeatureExtractor, PSDFeatureExtractor
)
from einops import rearrange

class ALFEE(nn.Module):
    """
    ALFEE (Adaptive Large Foundation model for EEG) 아키텍처의 완전한 구현.
    논문에 기술된 모든 핵심 컴포넌트를 포함합니다.
    """
    def __init__(self, num_classes, dim, num_channels, patch_size, 
                 channel_blocks, temporal_blocks, decoder_blocks, heads, 
                 msc_n_convs, msc_kernel_sizes, msc_strides, msc_out_channels,
                 psd_n_fft, psd_hop_length, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dim = dim

        # --- 1. Feature Extractor ---
        # 논문에 기술된 다중 스케일 1D CNN 특징 추출기
        self.msc_extractor = MultiScaleConvFeatureExtractor(
            in_channels=1, 
            n_convs=msc_n_convs, 
            kernel_sizes=msc_kernel_sizes,
            strides=msc_strides,
            out_channels=msc_out_channels
        )
        # 논문에 기술된 Power Spectral Density(PSD) 특징 추출기
        self.psd_extractor = PSDFeatureExtractor(n_fft=psd_n_fft, hop_length=psd_hop_length)

        # 추출된 두 특징을 결합하여 모델의 메인 차원(dim)으로 프로젝션
        feature_dim = msc_out_channels + (psd_n_fft // 2 + 1)
        self.feature_projection = nn.Linear(feature_dim, dim)
        
        # --- 2. Encoders & Decoder ---
        self.rotary_emb = RotaryEmbedding(dim // heads)
        
        # 채널 인코더: 채널 간의 공간적 관계를 학습
        self.channel_encoder = nn.ModuleList([
            TransformerBlock(dim, heads, is_cross_attention=True) for _ in range(channel_blocks)
        ])
        
        # 시간 인코더: 시간의 흐름에 따른 동적 관계를 학습
        self.temporal_encoder = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(temporal_blocks)
        ])

        # EEG 디코더: 인코딩된 특징으로부터 원본 신호를 복원
        self.decoder = nn.ModuleList([
            TransformerBlock(dim, heads, is_cross_attention=True) for _ in range(decoder_blocks)
        ])
        
        # --- 3. Learnable Tokens & Queries ---
        # 채널 인코딩을 위한 학습 가능한 쿼리
        self.channel_query = nn.Parameter(torch.randn(1, 1, dim))
        # 데이터셋 종류를 학습하기 위한 토큰 (L_DT 손실용)
        self.data_task_token = nn.Parameter(torch.randn(1, 1, dim))
        # 디코더가 재구성을 시작하기 위한 학습 가능한 쿼리
        self.decoder_query = nn.Parameter(torch.randn(1, num_channels, dim))

        # --- 4. Heads ---
        # 파인튜닝 시 사용할 태스크별 분류 헤드
        self.finetune_heads = nn.ModuleDict({
            task_name: nn.Linear(dim, n_class)
            for task_name, n_class in num_classes.items()
        })
        # 사전학습 시 신호 재구성을 위한 헤드
        self.reconstruction_head = nn.Linear(dim, patch_size)
        # L_DT 손실 계산을 위한 헤드
        self.task_classification_head = nn.Linear(dim, len(kwargs.get('task_type_map', {})))

    def forward(self, x, mode='finetune', task_name=None, mask_info=None, **kwargs):
        b, t, c, fs = x.shape
        x_reshaped = rearrange(x, 'b t c fs -> (b t c) 1 fs')

        # 1. 특징 추출
        msc_features = self.msc_extractor(x_reshaped)
        psd_features = self.psd_extractor(x_reshaped.squeeze(1))
        combined_features = torch.cat([msc_features, psd_features], dim=-1)
        projected_features = self.feature_projection(combined_features)
        
        # (B*T*C, Dim) -> (B*T, C, Dim)
        features = rearrange(projected_features, '(b t c) d -> (b t) c d', b=b, t=t, c=c)

        # 2. 채널 인코더
        channel_query_expanded = self.channel_query.expand(b * t, -1, -1)
        for block in self.channel_encoder:
            channel_query_expanded = block(channel_query_expanded, context=features)
        
        # (B*T, 1, Dim) -> (B, T, Dim)
        encoded_channel = rearrange(channel_query_expanded, '(b t) n d -> b (t n) d', b=b, t=t)
        
        # 3. 시간 인코더
        # 데이터셋 분류 토큰(DT) 추가
        dt_expanded = self.data_task_token.expand(b, -1, -1)
        temporal_input = torch.cat([dt_expanded, encoded_channel], dim=1)
        
        # 로터리 위치 임베딩 적용
        pos_emb = self.rotary_emb(temporal_input.shape[1], device=x.device)
        encoded_temporal = temporal_input
        for block in self.temporal_encoder:
            encoded_temporal = block(encoded_temporal, pos_emb=pos_emb)
            
        # DT 토큰 분리
        dt_output = encoded_temporal[:, 0]
        temporal_features = encoded_temporal[:, 1:]

        if mode == 'pretrain':
            # 4. EEG 디코더 (사전학습 시)
            decoder_query_expanded = self.decoder_query.expand(b * t, -1, -1)
            # 시간 특징을 디코더의 context로 사용
            decoder_context = temporal_features.reshape(b*t, 1, self.dim)
            
            decoded = decoder_query_expanded
            for block in self.decoder:
                decoded = block(decoded, context=decoder_context)
            
            # (B*T, C, Dim) -> (B, T, C, Fs)
            reconstructed_signal = self.reconstruction_head(decoded)
            reconstructed_signal = rearrange(reconstructed_signal, '(b t) c fs -> b t c fs', b=b, t=t)
            
            # 데이터셋 분류 로짓
            task_logits = self.task_classification_head(dt_output)
            
            return reconstructed_signal, temporal_features, task_logits
        else: # finetune
            # 5. 분류 헤드 (파인튜닝 시)
            pooled_features = temporal_features.mean(dim=1)
            
            if task_name not in self.finetune_heads:
                raise ValueError(f"Task '{task_name}' not found. Available: {list(self.finetune_heads.keys())}")
            
            logits = self.finetune_heads[task_name](pooled_features)

            # 파인튜닝 시 재구성 손실을 위해 디코더도 통과
            if self.training:
                # (재사용 로직)
                decoder_query_expanded = self.decoder_query.expand(b * t, -1, -1)
                decoder_context = temporal_features.reshape(b*t, 1, self.dim)
                decoded = decoder_query_expanded
                for block in self.decoder:
                    decoded = block(decoded, context=decoder_context)
                reconstructed_signal = self.reconstruction_head(decoded)
                reconstructed_signal = rearrange(reconstructed_signal, '(b t) c fs -> b t c fs', b=b, t=t)
                return logits, reconstructed_signal
            
            return logits
