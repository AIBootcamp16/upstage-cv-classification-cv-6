# CV 6조
## Keep Going!!


| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [문서연](https://github.com/UpstageAILab)             |            [최현](https://github.com/UpstageAILab)             |            [김수현](https://github.com/UpstageAILab)             |            [이승호](https://github.com/UpstageAILab)             |            [오정택](https://github.com/UpstageAILab)             |
|                            팀장, 실험총괄                             |                            ConvNext 모델링                             |                            ViT 모델링                             |                            EfficientNet_v2_s 실험                             |                         EfficientNet_v2_m 실험                            |

## 0. Overview
### Environment
<img width="322" height="188" alt="image" src="https://github.com/user-attachments/assets/97a3313b-36fe-4bc5-a77e-e75afa295b45" />


### Requirements
<img width="231" height="217" alt="image" src="https://github.com/user-attachments/assets/763e03cb-4beb-4ff1-87f4-bc31011a8c9c" />


## 1. Competiton Info

### Overview

이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용된 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

### Timeline

- 2025년 10월 31일 (금) 10:00 - Start Date
- 2025년 11월 12일 (수) 19:00 - Final submission deadline

## 2. Components

### Directory

<img width="322" height="361" alt="image" src="https://github.com/user-attachments/assets/5e352d8e-2783-4dd7-8d84-3a0e6788027c" />

## 3. Data descrption

### Dataset overview

<img width="374" height="138" alt="image" src="https://github.com/user-attachments/assets/7a7860c7-13c2-410d-bc60-c42674d8b703" />

#### 학습 데이터셋 정보
train [폴더] 1570장의 이미지가 저장되어 있습니다.
train.csv [파일] 1570개의 행으로 이루어져 있습니다. train/ 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.
- ID: 학습 샘플의 파일명
- target: 학습 샘플의 정답 클래스 번호

#### meta.csv [파일] 17개의 행으로 이루어져 있습니다.
- target: 17개의 클래스 번호입니다.
- class_name: 클래스 번호에 대응하는 클래스 이름입니다.

#### 평가 데이터셋 정보
test [폴더] 3140장의 이미지가 저장되어 있습니다.
- ID: 평가 샘플의 파일명이 저장되어 있습니다.
- target: 예측 결과가 입력될 컬럼입니다. 값이 전부 0으로 저장되어 있습니다.

### EDA

#### Ramdom Train images
<img width="5878" height="1600" alt="train_samples" src="https://github.com/user-attachments/assets/5b6e606c-6b40-4618-b86f-e87050b57a7b" />
<img width="5840" height="1600" alt="train_samples_set2" src="https://github.com/user-attachments/assets/9466f128-9533-4e3d-8fb2-60cecd7ae615" />
<img width="5840" height="1600" alt="train_samples_set3" src="https://github.com/user-attachments/assets/b19567d0-4a7e-4800-9db7-69d77f5f3604" />

#### Ramdom Test images
<img width="5738" height="1538" alt="test_samples" src="https://github.com/user-attachments/assets/55d3befe-bf00-48e4-a115-6e77ee04e104" />
<img width="5796" height="1561" alt="test_samples_set2" src="https://github.com/user-attachments/assets/ab36339a-4e26-4d3f-9616-d6d852da5889" />
<img width="5970" height="1599" alt="test_samples_set3" src="https://github.com/user-attachments/assets/ac57acb2-bd5c-44d4-bee0-914c7a7fb2a2" />

### 클래스별 분포 분석
<img width="1338" height="1284" alt="image" src="https://github.com/user-attachments/assets/5786969d-c3a4-4c6c-91e4-759fc2e6537f" />

## 4. Modeling

### Baseline code - Private Board 0.9282

<img width="708" height="1104" alt="image" src="https://github.com/user-attachments/assets/d65c7fdc-dd27-4f00-94b8-577839869739" />
<img width="722" height="1196" alt="image" src="https://github.com/user-attachments/assets/4bf8700b-0bb4-4aa5-9abb-1cd38f6dae80" />

## Modeling Process - Ensemble

## 사용된 모델

### 1. ViT (Vision Transformer) 계열
- **vit_base**: `vit_base_patch16_384`
  - Drop rate: 0.2, Drop path rate: 0.2
- **vit_base_strong**: `vit_base_patch16_384`
  - Drop rate: 0.3, Drop path rate: 0.3 (더 강한 regularization)
- **vit_large**: `vit_large_patch16_384`
  - Drop rate: 0.2, Drop path rate: 0.2

### 2. ConvNeXt 계열
- **convnext_large**: `convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384`
  - CLIP LAION2B pretrained + ImageNet-12k/1k fine-tuned
  - Drop rate: 0.2

### 3. Swin Transformer 계열
- **swin_large**: `swin_large_patch4_window12_384`
  - Drop rate: 0.2

모든 모델은 timm 라이브러리를 통해 pretrained weights 사용

---

## 앙상블 방법

### 1. OOF (Out-of-Fold) 기반 가중치 최적화

#### 최적화 프로세스
1. **Dirichlet 샘플링**:
   - `n_iter`(기본 200회) 동안 Dirichlet 분포에서 가중치 샘플링
   - OOF macro F1 score로 평가

2. **Capped Simplex Projection**:
   - `--weight-cap` 옵션으로 개별 모델 가중치 상한 제한
   - Bisection method로 simplex 제약 하에 투영

3. **Local Refinement** (Coordinate Ascent):
   - 50회 반복으로 각 가중치를 ±0.05씩 조정
   - 더 이상 개선 없을 때까지 반복

### 2. 앙상블 공간 선택

#### Probability Space (기본)
```python
blended = Σ(weight[i] * prob[i])
prediction = argmax(blended)
```

#### Logit Space
```python
blended_logits = Σ(weight[i] * logit[i])
blended_probs = softmax(blended_logits)
prediction = argmax(blended_probs)
```

### 3. 후처리 기법

#### Temperature Scaling (Calibration)
- OOF logits에 LBFGS 최적화로 temperature 파라미터 학습
- 모델 confidence 보정

#### Prior Alignment
- 학습 데이터 클래스 분포와 예측 분포 정렬
- 공식: `adjusted = predictions * (π_target / π_pred)^α`

---

## 학습 기법

### 1. Data Augmentation

#### Base Augmentation
- HorizontalFlip (p=0.5), VerticalFlip (p=0.5)
- Rotate (±180°, p=0.7)
- ShiftScaleRotate (p=0.6)
- 밝기/대비 또는 HSV 조정 (p=0.5)

#### Strong Augmentation
- Base 기법 + 더 강한 확률과 파라미터
- GaussNoise/Blur/MotionBlur (p=0.5)
- CoarseDropout (p=0.5)
- Grid/Optical Distortion (p=0.3)

### 2. Mixup & CutMix
- Alpha=0.4, CutMix 확률 50%
- `--mixup-warmup` epoch 이후 적용
- Mixed criterion으로 손실 계산

### 3. Regularization
- Label smoothing (기본 0.1)
- Gradient clipping (max_norm=1.0)
- Class-balanced weighted loss
- Dropout & DropPath (모델별 설정)

### 4. 최적화
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)
- Mixed Precision Training (AMP)
- Gradient Accumulation

---

## TTA (Test-Time Augmentation)

### 변환 종류
- `original`: 원본
- `hflip`: 수평 뒤집기
- `vflip`: 수직 뒤집기
- `rotate90`: 90도 회전

### Multi-Scale TTA
- ViT 계열: 고정 384 (architecture 제약)
- 기타 모델: 설정된 scales (예: 352, 384, 416)

### 집계 방법
- 각 fold × TTA 변환 × scales 조합의 예측 평균
- 최종: fold 평균

---

## 5. Result

### Leader Board

<img width="967" height="309" alt="image" src="https://github.com/user-attachments/assets/c56e65e6-2fc2-4503-b1b5-a09cd7cf0775" />

6조 3위 0.9401

### Reference

- https://www.kaggle.com/
- https://huggingface.co/timm
