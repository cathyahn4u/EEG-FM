import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset, IterableDataset, random_split
from data_handling import datasets as dataset_classes
from data_handling.preprocessing import PREPROCESSING_FN_MAP, preprocess_for_labram_original
from transformers import AutoTokenizer

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config['model_selection']
        self.data_conf = config['data']
        self.dataset_conf = config['datasets']
        self.training_conf = config['training']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['NeuroLM']['params']['llm_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_preprocessing_fn(self):
        is_pretrain = self.trainer.model.is_pretrain if self.trainer else True
        if self.model_name == 'CBraMod':
            if (not is_pretrain and self.training_conf['finetune']['original_ckpt_path']) or \
               (is_pretrain and self.training_conf['pretrain']['cbramod_mode'] == 'supervised'):
                return PREPROCESSING_FN_MAP['cbramod_original']
        return PREPROCESSING_FN_MAP['standard']

    def setup(self, stage: str):
        self.preproc_fn = self._get_preprocessing_fn()
        
        if stage == 'fit':
            if (self.training_conf['pretrain']['cbramod_mode'] == 'supervised' and self.model_name in ['CBraMod', 'NeuroLM']) \
            or (self.model_name == 'NeuroLM'): # NeuroLM is always supervised
                pretrain_list = self.dataset_conf['lists']['supervised_pretrain']
            else:
                pretrain_list = self.dataset_conf['lists']['self_supervised_pretrain']
            
            pretrain_datasets = self._create_datasets(pretrain_list)
            self.train_pretrain_dataset = ConcatDataset(pretrain_datasets) if pretrain_datasets else None

            finetune_list = self.dataset_conf['lists']['finetune']
            finetune_datasets = self._create_datasets(finetune_list)
            if finetune_datasets:
                full_finetune_dataset = ConcatDataset(finetune_datasets)
                train_len = int(len(full_finetune_dataset) * 0.8)
                val_len = len(full_finetune_dataset) - train_len
                self.train_finetune_dataset, self.val_finetune_dataset = random_split(full_finetune_dataset, [train_len, val_len])
            else:
                self.train_finetune_dataset, self.val_finetune_dataset = None, None

        elif stage == 'test':
            test_list = self.dataset_conf['lists']['finetune']
            test_datasets = self._create_datasets(test_list)
            self.test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    def _create_datasets(self, dataset_names):
        datasets_list = []
        for name in dataset_names:
            if name in self.dataset_conf['properties']:
                props = self.dataset_conf['properties'][name]
                try:
                    # GenericEEGDataset을 상속받는 클래스 동적 생성
                    DatasetClass = dataset_classes.create_dataset_class(f"{name}Dataset", name, props['task_type'])
                    dataset = DatasetClass(path=props['path'], target_fs=self.data_conf['fs'])
                    dataset.preprocessing_fn = self.preproc_fn
                    datasets_list.append(dataset)
                except Exception as e:
                    print(f"Warning: Failed to create dataset '{name}'. {e}")
        return datasets_list

    def _collate_fn(self, batch):
        batch = [b for b in batch if b is not None and b['signal'] is not None and b['signal'].shape[0] > 0]
        if not batch: return None

        signals, labels, task_names, metadata_list = [], [], [], []
        for b in batch:
            signals.append(b['signal'])
            labels.append(torch.full((b['signal'].shape[0],), b['label']))
            task_names.extend([b['task_name']] * b['signal'].shape[0])
            metadata_list.extend([b['metadata']] * b['signal'].shape[0])

        signals = torch.cat(signals, dim=0)
        labels = torch.cat(labels, dim=0)

        processed_signals = signals
        if self.model_name in ['LaBraM', 'NeuroLM']:
            signals_for_stft = rearrange(signals, 'b t c l -> (b t) c l')
            processed_signals = preprocess_for_labram_original(signals_for_stft)

        texts = [m.get('text', '') for m in metadata_list]
        tokenized_text = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)

        return processed_signals, labels, task_names, tokenized_text, metadata_list

    def train_dataloader(self):
        is_pretrain = self.trainer.model.is_pretrain
        dataset = self.train_pretrain_dataset if is_pretrain else self.train_finetune_dataset
        batch_size = self.training_conf['pretrain']['batch_size'] if is_pretrain else self.training_conf['finetune']['batch_size']
        if dataset is None: return None
        return DataLoader(dataset, batch_size=batch_size, shuffle=not isinstance(dataset, IterableDataset), num_workers=self.data_conf['num_workers'], collate_fn=self._collate_fn, drop_last=True, persistent_workers=True if self.data_conf['num_workers'] > 0 else False)

    def val_dataloader(self):
        if self.val_finetune_dataset is None: return None
        return DataLoader(self.val_finetune_dataset, batch_size=self.training_conf['finetune']['batch_size'], num_workers=self.data_conf['num_workers'], collate_fn=self._collate_fn, persistent_workers=True if self.data_conf['num_workers'] > 0 else False)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.training_conf['finetune']['batch_size'], num_workers=self.data_conf['num_workers'], collate_fn=self._collate_fn)
