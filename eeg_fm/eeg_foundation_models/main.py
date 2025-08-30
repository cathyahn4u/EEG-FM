import argparse
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data_handling.eeg_datamodule import EEGDataModule
from lightning.eeg_lightning_module import EEGFoundationLightningModule

def main(args):
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    if args.model: config['model_selection'] = args.model
    if args.checkpoint_path: config['training']['finetune']['checkpoint_path'] = args.checkpoint_path
    if args.original_ckpt_path: config['training']['finetune']['original_ckpt_path'] = args.original_ckpt_path
    
    model_name = config['model_selection']
    is_pretrain = (args.mode == 'pretrain')
    
    datamodule = EEGDataModule(config)
    model = EEGFoundationLightningModule(config, is_pretrain=is_pretrain)

    callbacks = []
    if not is_pretrain:
        monitor_task = config["datasets"]["lists"]["finetune"][0]
        monitor_metric = f'val/{monitor_task}_loss'
        callbacks.extend([
            ModelCheckpoint(
                dirpath=f'checkpoints/{model_name}/finetune/',
                filename=f'{{epoch}}-{{{monitor_metric}:.2f}}',
                save_top_k=1, verbose=True, monitor=monitor_metric, mode='min'
            ),
            EarlyStopping(
                monitor=monitor_metric, patience=config['training']['common']['patience'],
                verbose=True, mode='min'
            )
        ])

    trainer = pl.Trainer(
        max_epochs=config['training']['pretrain' if is_pretrain else 'finetune']['epochs'],
        accelerator=config['training']['common']['accelerator'],
        devices=config['training']['common']['devices'],
        precision=config['training']['common']['precision'],
        callbacks=callbacks or None,
        limit_val_batches=0.0 if is_pretrain else 1.0
    )

    if args.mode == 'pretrain':
        trainer.fit(model, datamodule=datamodule)
    elif args.mode == 'finetune':
        ckpt_path = config['training']['finetune']['checkpoint_path']
        original_ckpt_path = config['training']['finetune']['original_ckpt_path']
        
        if original_ckpt_path: model.load_from_original_ckpt(original_ckpt_path, model_name)
        
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path if not original_ckpt_path else None)

    elif args.mode == 'evaluate':
        ckpt_path = config['training']['finetune']['checkpoint_path']
        if not ckpt_path: raise ValueError("Checkpoint path must be provided for evaluation.")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune', 'evaluate'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--original_ckpt_path', type=str, default=None)
    main(parser.parse_args())
