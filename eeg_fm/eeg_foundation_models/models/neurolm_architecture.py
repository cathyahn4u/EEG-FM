import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from models.labram_architecture import LaBraM

class NeuroLM(nn.Module):
    def __init__(self, num_classes, eeg_encoder_params, llm_name, projection_dim, **kwargs):
        super().__init__()
        self.eeg_encoder = LaBraM(num_classes=1, **eeg_encoder_params)
        self.llm_config = AutoConfig.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.projection = nn.Linear(self.eeg_encoder.d_model, projection_dim)
        n_class = next(iter(num_classes.values())) if isinstance(num_classes, dict) else num_classes
        self.classification_head = nn.Linear(self.llm_config.n_embd, n_class)

    def forward(self, eeg_signal_stft, tokenized_text, **kwargs):
        eeg_features = self.eeg_encoder(eeg_signal_stft, mode='finetune', return_features_only=True)
        projected_eeg = self.projection(eeg_features).unsqueeze(1)
        
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        text_embeds = self.llm.transformer.wte(input_ids)
        
        combined_embeds = torch.cat([projected_eeg, text_embeds], dim=1)
        combined_attention_mask = torch.cat([torch.ones_like(projected_eeg[:, :, 0]), attention_mask], dim=1)
        
        outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask, output_hidden_states=True)
        logits = self.classification_head(outputs.hidden_states[-1][:, -1, :])
        return logits

    @property
    def device(self):
        return next(self.parameters()).device
