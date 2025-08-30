import torch
import lightning.pytorch as pl
import yaml
import tempfile
import os
from torch.utils.data import DataLoader, Dataset
from elightning.eeg_lightning_module import EEGFoundationLightningModule
from transformers import AutoTokenizer
from einops import rearrange
from data_handling.preprocessing import preprocess_for_labram_original

class DummyEEGDataset(Dataset):
    def __init__(self, num_samples, num_channels, seq_len_points, num_classes, task_name, task_type, has_text=False):
        self.num_samples, self.num_channels, self.seq_len_points = num_samples, num_channels, seq_len_points
        self.num_classes, self.task_name, self.task_type, self.has_text = num_classes, task_name, task_type, has_text
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        signal = torch.randn(1, self.num_channels, self.seq_len_points)
        label = torch.randint(0, self.num_classes, (1,)).item()
        metadata = {"text": f"Dummy text {idx}", "task_type": self.task_type} if self.has_text else {"task_type": self.task_type}
        return {"signal": signal, "label": label, "task_name": self.task_name, "metadata": metadata}

def get_collate_fn(tokenizer, model_name):
    def collate_fn(batch):
        signals = torch.cat([b['signal'] for b in batch], dim=0)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        task_names = [b['task_name'] for b in batch]
        metadata_list = [b['metadata'] for b in batch]
        
        processed_signals = signals
        if model_name in ['LaBraM', 'NeuroLM']:
            processed_signals = preprocess_for_labram_original(rearrange(signals, 'b t c l -> (b t) c l'))

        texts = [m.get('text', '') for m in metadata_list]
        tokenized = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        return processed_signals, labels, task_names, tokenized, metadata_list
    return collate_fn

def run_pretrain_finetune_flow(config, model_name, mode, tmpdir, accelerator, collate_fn):
    print(f"\n🚀 [{model_name}{'-'+mode if mode!='default' else ''}] Phase 1/2: Dummy Pre-training...")
    config['training']['finetune']['original_ckpt_path'] = None # Clear original ckpt path for this flow
    config['training']['finetune']['checkpoint_path'] = None # Clear project ckpt path
    
    pretrain_model = EEGFoundationLightningModule(config, is_pretrain=True)
    
    is_supervised = (model_name == 'CBraMod' and mode == 'supervised') or model_name == 'NeuroLM'
    pretrain_task_name = config['datasets']['lists']['supervised_pretrain'][0] if is_supervised else config['datasets']['lists']['self_supervised_pretrain'][0]
    props = config['datasets']['properties'][pretrain_task_name]
    
    num_channels = config['model'][model_name]['params'].get('num_channels', 90)
    pretrain_dataset = DummyEEGDataset(16, num_channels, 256*4, props['num_classes'], pretrain_task_name, props['task_type'], has_text=(model_name == 'NeuroLM'))
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=4, collate_fn=collate_fn)
    
    trainer = pl.Trainer(max_epochs=2, accelerator=accelerator, devices=1, default_root_dir=tmpdir, logger=False, enable_checkpointing=True, limit_val_batches=0)
    trainer.fit(pretrain_model, train_dataloaders=pretrain_loader)
    
    checkpoint_path = trainer.checkpoint_callback.best_model_path or os.path.join(tmpdir, "temp_ckpt.ckpt")
    if not os.path.exists(checkpoint_path): trainer.save_checkpoint(checkpoint_path)
    print(f"✅ Pre-training successful.")

    print(f"\n🚀 [{model_name}{'-'+mode if mode!='default' else ''}] Phase 2/2: Dummy Fine-tuning...")
    
    finetune_task_name = config['datasets']['lists']['finetune'][0]
    ft_props = config['datasets']['properties'][finetune_task_name]
    
    finetune_dataset = DummyEEGDataset(16, num_channels, 256*4, ft_props['num_classes'], finetune_task_name, ft_props['task_type'], has_text=(model_name == 'NeuroLM'))
    finetune_loader = DataLoader(finetune_dataset, batch_size=4, collate_fn=collate_fn)

    # Test both loading methods
    # 1. From original .pth
    if model_name in ['CBraMod', 'LaBraM', 'NeuroLM']:
        print("  - Testing fine-tuning from original .pth checkpoint...")
        dummy_original_path = os.path.join(tmpdir, f"original_{model_name}_{mode}.pth")
        torch.save(pretrain_model.model.state_dict(), dummy_original_path)
        config['training']['finetune']['original_ckpt_path'] = dummy_original_path
        finetune_model_orig = EEGFoundationLightningModule(config, is_pretrain=False)
        finetune_model_orig.load_from_original_ckpt(dummy_original_path, model_name)
        trainer_ft_orig = pl.Trainer(max_epochs=2, accelerator=accelerator, devices=1, default_root_dir=tmpdir, logger=False, enable_checkpointing=False)
        trainer_ft_orig.fit(finetune_model_orig, train_dataloaders=finetune_loader, val_dataloaders=finetune_loader)

    # 2. From project .ckpt
    print("  - Testing fine-tuning from project .ckpt checkpoint...")
    config['training']['finetune']['original_ckpt_path'] = None
    finetune_model_ckpt = EEGFoundationLightningModule.load_from_checkpoint(checkpoint_path, config=config, is_pretrain=False, strict=False)
    trainer_ft_ckpt = pl.Trainer(max_epochs=2, accelerator=accelerator, devices=1, default_root_dir=tmpdir, logger=False, enable_checkpointing=False)
    trainer_ft_ckpt.fit(finetune_model_ckpt, train_dataloaders=finetune_loader, val_dataloaders=finetune_loader)
    
    print(f"✅ Fine-tuning successful.")

def run_dummy_test_for_model(model_name):
    print("\n" + "="*20 + f" RUNNING DUMMY TEST FOR: {model_name} " + "="*20)
    
    with open('configs/default_config.yaml', 'r') as f: config = yaml.safe_load(f)
    config['model_selection'] = model_name

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Dummy test will run on: {accelerator.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(config['model']['NeuroLM']['params']['llm_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    collate_fn = get_collate_fn(tokenizer, model_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        if model_name == 'CBraMod':
            config['training']['pretrain']['cbramod_mode'] = 'supervised'
            run_pretrain_finetune_flow(config, model_name, 'supervised', tmpdir, accelerator, collate_fn)
            config['training']['pretrain']['cbramod_mode'] = 'self-supervised'
            run_pretrain_finetune_flow(config, model_name, 'self-supervised', tmpdir, accelerator, collate_fn)
        else:
            run_pretrain_finetune_flow(config, model_name, 'default', tmpdir, accelerator, collate_fn)

if __name__ == '__main__':
    run_dummy_test_for_model('ALFEE')
    run_dummy_test_for_model('CBraMod')
    run_dummy_test_for_model('LaBraM')
    run_dummy_test_for_model('NeuroLM')
