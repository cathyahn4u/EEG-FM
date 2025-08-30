import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score, AUROC
from models.alfee_architecture import ALFEE
from models.cbramod_architecture import CBraMod
from models.labram_architecture import LaBraM
from models.neurolm_architecture import NeuroLM
from einops import rearrange

class EEGFoundationLightningModule(pl.LightningModule):
    def __init__(self, config, is_pretrain=True):
        super().__init__()
        self.save_hyperparameters()
        self.config, self.is_pretrain = config, is_pretrain
        self.model_name = config['model_selection']
        self.model_params = config['model'][self.model_name]['params']
        self.training_params = config['training']
        self.task_type_map = config['datasets']['task_type_map']

        finetune_tasks = {n: p['num_classes'] for n, p in config['datasets']['properties'].items() if n in config['datasets']['lists']['finetune']}
        self._init_model(finetune_tasks)
        self._init_losses_and_metrics(finetune_tasks)

    def _init_model(self, num_classes):
        model_map = {'ALFEE': ALFEE, 'CBraMod': CBraMod, 'LaBraM': LaBraM, 'NeuroLM': NeuroLM}
        self.model_params['task_type_map'] = self.task_type_map
        self.model = model_map[self.model_name](num_classes=num_classes, **self.model_params)

    def _init_losses_and_metrics(self, tasks):
        self.loss_mse, self.loss_ce = nn.MSELoss(), nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            name: nn.ModuleDict({
                'acc': Accuracy(task="multiclass", num_classes=n_class, top_k=1),
                'f1': F1Score(task="multiclass" if n_class > 2 else "binary", num_classes=n_class),
                'auroc': AUROC(task="binary") if n_class == 2 else nn.Module()
            }) for name, n_class in tasks.items()
        })
    
    def load_state_dict(self, state_dict, strict=True):
        if not self.is_pretrain and not strict:
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            if not pretrained_dict: print("Warning: No matching layers found.")
            model_dict.update(pretrained_dict)
            missing, unexpected = super().load_state_dict(model_dict, strict=False)
            if not unexpected: print("✅ Loaded pre-trained weights, ignoring mismatched layers.")
            return missing, unexpected
        return super().load_state_dict(state_dict, strict=strict)

    def load_from_original_ckpt(self, ckpt_path, model_name):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        target_model = self.model.eeg_encoder if model_name == 'NeuroLM' else self.model
        if next(iter(state_dict)).startswith('module.'): state_dict = {k[7:]: v for k, v in state_dict.items()}
        target_model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from original {model_name} checkpoint.")

    def training_step(self, batch, batch_idx):
        if not batch: return None
        loss = self._calculate_pretrain_loss(batch) if self.is_pretrain else self._calculate_finetune_loss(batch)
        stage = "pretrain" if self.is_pretrain else "train"
        self.log(f"{stage}/{batch[2][0]}_loss", loss, prog_bar=True)
        return loss

    def _calculate_pretrain_loss(self, batch):
        signals, labels, task_names, tokenized_text, metadata = batch
        if self.model_name == 'ALFEE':
            reconstructed, _, task_logits = self.model(signals, mode='pretrain')
            task_types = torch.tensor([self.task_type_map[m['task_type']] for m in metadata], device=self.device)
            loss_rec = self.loss_mse(reconstructed, signals) # L_GPT, L_MAE 통합
            loss_dt = self.loss_ce(task_logits, task_types)
            return loss_rec + self.training_params['pretrain']['lambda_dt'] * loss_dt
        elif self.model_name == 'CBraMod':
            if self.training_params['pretrain']['cbramod_mode'] == 'supervised':
                return self.loss_ce(self.model(signals, mode='finetune'), labels)
            else:
                b, t, c, fs = signals.shape
                mask = (torch.rand(b, self.model.num_channels, device=self.device) > self.training_params['mask_ratio'])
                reconstructed = self.model(signals * mask.view(b, 1, c, 1).float(), mode='pretrain_ssl')
                original_patches = torch.mean(self.model.patch_embed(rearrange(signals, 'b t c fs -> b c (t fs)')), dim=2)
                return self.loss_mse(reconstructed, original_patches)
        elif self.model_name == 'LaBraM':
            mask = (torch.rand(signals.shape[0], self.model_params['num_channels'], device=self.device) > self.training_params['mask_ratio'])
            reconstructed, original, commit_loss = self.model(signals, mode='pretrain', mask=mask)
            return self.loss_mse(reconstructed, original) + commit_loss
        elif self.model_name == 'NeuroLM':
            return self.loss_ce(self.model(signals, tokenized_text), labels)
        return torch.tensor(0.0, device=self.device)

    def _calculate_finetune_loss(self, batch):
        signals, labels, task_names, tokenized_text, metadata = batch
        task = task_names[0]
        output = self.model(signals, mode='finetune', task_name=task, tokenized_text=tokenized_text)
        if self.model_name == 'ALFEE' and isinstance(output, tuple):
            logits, reconstructed = output
            loss_cls = self.loss_ce(logits, labels)
            loss_rec = self.loss_mse(reconstructed, signals)
            return self.training_params['finetune']['alpha_cls_loss'] * loss_cls + (1 - self.training_params['finetune']['alpha_cls_loss']) * loss_rec
        return self.loss_ce(output, labels)

    def _shared_eval_step(self, batch, stage):
        if not batch: return
        signals, labels, task_names, tokenized_text, metadata = batch
        task = task_names[0]
        output = self.model(signals, mode='finetune', task_name=task, tokenized_text=tokenized_text)
        logits = output[0] if isinstance(output, tuple) else output
        loss = self.loss_ce(logits, labels)
        self.log(f"{stage}/{task}_loss", loss, prog_bar=True)
        if task in self.metrics:
            for name, metric in self.metrics[task].items():
                if isinstance(metric, nn.Module) and not list(metric.children()): continue
                metric.update(logits, labels)
                self.log(f"{stage}/{task}_{name}", metric, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        if not self.is_pretrain: self._shared_eval_step(batch, 'val')
    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, 'test')
    def configure_optimizers(self):
        lr = self.training_params['pretrain' if self.is_pretrain else 'finetune']['learning_rate']
        return torch.optim.AdamW(self.parameters(), lr=lr)
