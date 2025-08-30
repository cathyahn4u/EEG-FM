# EEG Foundation Models Project (ALFEE + CBraMod + LaBraM + NeuroLM)

이 프로젝트는 ALFEE, CBraMod, LaBraM, NeuroLM 네 가지 주요 EEG Foundation Model을 통합하여, 동일한 데이터 처리 파이프라인과 학습 프레임워크 내에서 선택적으로 사용할 수 있도록 리팩토링한 최종 버전입니다.

## ✨ 주요 특징

- **4개 모델 완벽 지원**: `config.yaml` 파일 수정만으로 **ALFEE**, **CBraMod**, **LaBraM**, **NeuroLM** 모델을 손쉽게 교체하여 학습 및 평가할 수 있습니다.
- **모델별 고유 학습 방식 지원**:
  - **ALFEE**: Multi-Scale Conv, PSD 특징 추출, 4가지 손실 함수 등 논문의 모든 핵심 기능을 포함한 Self-Supervised Pre-training -> Fine-tuning.
  - **CBraMod**: Supervised 및 Self-Supervised Pre-training -> Fine-tuning. **Criss-Cross Attention** 및 **분류기 옵션** 지원.
  - **LaBraM**: VQ-VAE 기반 채널 마스킹 Self-Supervised Pre-training -> Fine-tuning.
  - **NeuroLM**: EEG-Language Multi-modal Supervised Pre-training -> Fine-tuning.
- **사전 학습된 가중치 재활용**: CBraMod, LaBraM, NeuroLM 원본 GitHub의 **`.pth` 가중치를 불러와 파인튜닝**하는 기능을 지원합니다.
- **공용화된 전처리 파이프라인**: 모델별/데이터셋별 전처리 로직을 공용화하여 **동일한 데이터셋으로 여러 모델을 쉽게 실험**할 수 있습니다.
- **Multi-modal 데이터 확장성**: 데이터 파이프라인이 EEG 신호 외에 텍스트, 타임스탬프 등 추가적인 메타데이터를 처리할 수 있도록 설계되었습니다.
- **최신 Lightning 프레임워크**: `pytorch_lightning` 대신 최신 `lightning.pytorch`를 사용합니다.
- **최종 검증 스크립트**: `dummy_train_test.py`를 통해 네 가지 모델 각각의 고유한 전체 학습/평가 파이프라인을 모두 검증할 수 있습니다.

## 📂 프로젝트 구조

```
eeg_foundation_models/
├── configs/
│   └── default_config.yaml
├── data_handling/
│   ├── base_dataset.py
│   ├── datasets.py
│   ├── eeg_datamodule.py
│   └── preprocessing.py
├── models/
│   ├── alfee_architecture.py
│   ├── cbramod_architecture.py
│   ├── labram_architecture.py
│   ├── neurolm_architecture.py
│   └── components.py
├── elightning/
│   └── eeg_lightning_module.py
├── main.py
├── dummy_train_test.py
├── requirements.txt
└── README.md
```

## 🚀 실행 방법

### 1. 모델 및 학습 방식 선택
`configs/default_config.yaml` 파일을 열고 `model_selection` 및 관련 학습 모드 값을 설정합니다.

### 2. 더미 데이터로 전체 워크플로우 검증 (강력 추천)
모든 학습 파이프라인이 정상 동작하는지 다음 명령어로 즉시 확인할 수 있습니다.
```bash
python dummy_train_test.py
```

### 3. 실제 데이터로 학습 및 평가
```bash
# 사전학습 (모델별 고유 방식 자동 적용)
python main.py --mode pretrain --model [MODEL_NAME]

# 이 프로젝트에서 사전학습한 모델로 파인튜닝
python main.py --mode finetune --model [MODEL_NAME] --checkpoint_path "path/to/pretrained.ckpt"

# 원본 가중치로 파인튜닝
python main.py --mode finetune --model [MODEL_NAME] --original_ckpt_path "path/to/original.pth"
```
