import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import urllib.request
import glob
import warnings
import wandb
import random 

# 경고 메시지 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. 설정 및 환경
# ==============================================================================

# WandB 로그인 (로그인 후 실행하세요)
# try:
#     wandb.login()
# except Exception as e:
#     print(f"WandB 로그인 오류: {e}. WandB 없이 진행됩니다.")

# Augraphy 라이브러리 체크 (필요시 설치)
try:
    from augraphy import InkBleed, PaperFactory, DirtyDrum, Jpeg, Brightness, AugraphyPipeline
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("⚠️ Augraphy 라이브러리를 찾을 수 없습니다. (pip install augraphy) - 문서 특화 증강 없이 진행됩니다.")

class Config:
    """우승을 목표로 하는 최종 하이퍼파라미터 설정 (EfficientNetV2-L + Mixup/CutMix)"""
    PROJECT_NAME = "document-classification-ultimate"
    RUN_NAME = "EffNetV2L_MixCut_A384_WandB"
    
    # 🔥 가장 강력한 EfficientNetV2 모델 사용
    MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k' 
    NUM_CLASSES = 17
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 🔥 Mixup/CutMix 통합 적용 (강력한 정규화)
    USE_MIX_STRATEGY = True
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0 # CutMix는 1.0이 일반적
    
    # 하이퍼파라미터 
    N_EPOCHS =35 # 충분한 학습을 위해 Epoch 증가
    BATCH_SIZE = 8  
    # 🔥 최적 해상도 384x384
    IMAGE_SIZE = 384  
    LR = 5e-5
    
    # 경사 누적 설정 (Effective Batch Size = 8 * 4 = 32 유지)
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # 정규화 및 최적화
    N_FOLDS = 5 
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.05
    
    # 스케줄러/정지
    SCHEDULER_T0 = 15 
    SCHEDULER_TMULT = 2
    PATIENCE = 7 
    
    # 경로 및 증강
    DATA_DIR = 'data'
    # 🔥 앙상블을 위해 모든 모델을 찾을 상위 폴더 (V1/V2/V3 모델 저장 경로)
    ENSEMBLE_MODEL_BASE_DIR = './experiments'


# ==============================================================================
# 2. 데이터 처리 함수 및 클래스
# ==============================================================================

def download_and_extract_data(data_dir='data'):
    """데이터 자동 다운로드 및 압축 해제"""
    data_path = Path(data_dir)
    
    if data_path.exists() and (data_path / 'train.csv').exists():
        print("✅ 데이터가 이미 존재합니다. 다운로드 건너김.\n")
        return
    
    print("="*70)
    print("📥 데이터 다운로드 및 준비 중...")
    print("="*70)
    
    DATA_URL = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000372/data/data.tar.gz"
    DATA_FILE = "data.tar.gz"
    
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        import tarfile
        with tarfile.open(DATA_FILE, 'r:gz') as tar:
            tar.extractall('.')
        os.remove(DATA_FILE)
        
        print("✅ 데이터 준비 완료!\n")
        
    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        raise

class DocumentDataset(Dataset):
    """문서 이미지 로드 및 라벨링을 위한 PyTorch Dataset"""
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx]['target']
            return image, torch.tensor(label, dtype=torch.long)

def ensure_3_channels(image):
    """Augraphy 후 2D로 변환되는 경우를 대비해 3채널(HxWx3)을 강제합니다."""
    if image.ndim == 2:
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        return image
    else:
        return image

def get_augraphy_pipeline(strategy='hybrid'):
    """문서 이미지 특화 증강 파이프라인"""
    if not AUGRAPHY_AVAILABLE:
        return None

    ink_p, paper_p, post_p = 0.7, 0.6, 0.5 

    ink_phase = [InkBleed(intensity_range=(0.05, 0.20), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.7)]
    post_phase = [Jpeg(quality_range=(50, 95), p=post_p), Brightness(brightness_range=(0.8, 1.2), p=post_p)]

    return AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)


def get_transforms(stage, cfg):
    """RandAugment 효과를 모방한 강력한 증강 및 384x384 리사이즈 적용"""
    augraphy_pipeline = get_augraphy_pipeline()
    
    if stage == 'train':
        albu_list = [
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), 
            # ShiftScaleRotate 강도 감소 (과도한 기하학적 변형 방지)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8, p=0.7), 
            A.HorizontalFlip(p=0.5),
            # RandAugment 효과를 위한 다양한 색상/노이즈 변환 통합
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
                A.GaussNoise(p=1),
            ], p=0.7),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, p=0.3), # Cutout 효과
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        if AUGRAPHY_AVAILABLE and augraphy_pipeline is not None:
            albu_list.insert(1, A.Lambda(
                image=lambda x, **kwargs: ensure_3_channels(augraphy_pipeline.augment(x)['output']), 
                p=1.0)
            )
            
        return A.Compose(albu_list)
    
    elif stage == 'val' or stage == 'test':
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return None

# ==============================================================================
# 3. 학습 및 추론 로직
# ==============================================================================

# 🔥 Mixup / CutMix 전략 함수
def mixup_cutmix_data(images, labels, alpha, mix_strategy='mixup'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    rand_index = torch.randperm(images.size(0)).to(images.device)
    
    # Mixup과 CutMix 로직을 단순화하여, 두 전략 모두 Logit Mixing을 사용합니다.
    mixed_images = lam * images + (1 - lam) * images[rand_index, :]

    label_a, label_b = labels, labels[rand_index]
    return mixed_images, label_a, label_b, lam

def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    """단일 Fold 학습 및 최적 모델 저장 (WandB, Mixup/CutMix 적용)"""
    print(f'\n{"="*50}')
    print(f'⚡️ Fold {fold} 학습 시작 - Model: {cfg.MODEL_NAME}')
    print(f'{"="*50}')
    
    # WandB 초기화 시도
    run = None
    try:
        run = wandb.init(project=cfg.PROJECT_NAME, name=f"{cfg.RUN_NAME}_Fold_{fold}", config=vars(cfg), reinit=True)
    except Exception as e:
        print(f"WandB 초기화 실패: {e}. 로깅 없이 진행됩니다.")
    
    train_dataset = DocumentDataset(train_df, f'{cfg.DATA_DIR}/train', get_transforms('train', cfg))
    val_dataset = DocumentDataset(val_df, f'{cfg.DATA_DIR}/train', get_transforms('val', cfg))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES)
    model.to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE), 
                                    label_smoothing=cfg.LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT, eta_min=1e-7
    )
    
    best_f1 = 0.0
    patience_counter = 0
    model_path = os.path.join(exp_dir, f'best_model_fold_{fold}.pth')
    
    for epoch in range(cfg.N_EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds_list, train_labels_list = [], []
        
        optimizer.zero_grad() 
        
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f'Fold {fold} | Epoch {epoch+1}/{cfg.N_EPOCHS} (Train)')):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            # 🔥 Mixup/CutMix 통합 적용 로직
            if cfg.USE_MIX_STRATEGY and random.random() < 0.8: # 80% 확률로 Mix
                strategy = 'mixup' if random.random() < 0.5 else 'cutmix' # 50% 확률로 Mixup/CutMix 선택
                alpha = cfg.MIXUP_ALPHA if strategy == 'mixup' else cfg.CUTMIX_ALPHA
                
                mixed_images, label_a, label_b, lam = mixup_cutmix_data(images, labels, alpha, strategy)
                outputs = model(mixed_images)
                # Mixup Loss
                loss = lam * criterion(outputs, label_a) + (1 - lam) * criterion(outputs, label_b)
                
                preds = outputs.argmax(dim=1) 
                target_labels = label_a
            else: # 일반 학습
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                target_labels = labels
            
            # 경사 누적
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step(epoch + (step + 1) / len(train_loader)) 
                optimizer.zero_grad() 
            
            running_loss += loss.item() * images.size(0) * cfg.GRADIENT_ACCUMULATION_STEPS
            train_preds_list.extend(preds.cpu().numpy())
            train_labels_list.extend(target_labels.cpu().numpy())
        
        if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
             optimizer.step()
             optimizer.zero_grad()

        epoch_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels_list, train_preds_list, average='macro')
        
        # Validation
        model.eval()
        val_preds_list, val_labels_list = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Fold {fold} | Epoch {epoch+1}/{cfg.N_EPOCHS} (Val)'):
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels_list, val_preds_list, average='macro')
        
        print(f"  [Result] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} (Best: {best_f1:.4f})")
        
        # WandB 로깅
        if run:
            run.log({
                "Fold": fold, "Epoch": epoch, "LR": scheduler.get_last_lr()[0],
                "Train/Loss": epoch_loss, "Train/F1": train_f1,
                "Val/Loss": val_loss, "Val/F1": val_f1
            })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  🏆 New Best F1: {best_f1:.4f}. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"  🛑 Early stopping on Fold {fold} at Epoch {epoch+1}")
                break
                
    if run:
        run.finish()
    return best_f1, model_path

@torch.no_grad()
def ultimate_inference_ensemble(cfg):
    """🔥 V1, V2, V3 모든 모델을 통합한 이종 모델 앙상블 추론 (PL 점수 0.9 목표)"""
    print('\n' + '='*70)
    print('🚀 최종 우승 앙상블 추론 시작 (모든 아키텍처 모델 통합)')
    print('='*70)

    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    test_df = test_df.drop('target', axis=1, errors='ignore')
    
    # 🔥 1. 모든 저장된 모델 경로 검색 (V1, V2, V3 실행 결과 포함)
    all_model_paths = glob.glob(os.path.join(cfg.ENSEMBLE_MODEL_BASE_DIR, '*', 'best_model_fold_*.pth'))
    
    if not all_model_paths:
        raise FileNotFoundError(f"'{cfg.ENSEMBLE_MODEL_BASE_DIR}' 폴더에서 모델 파일이 발견되지 않았습니다. 모든 전략(V1, V2, V3)을 실행하고 이 코드를 재실행하세요.")
    
    print(f"앙상블에 사용할 총 모델 수: {len(all_model_paths)}개")
    
    # 모델 아키텍처 이름 매핑 (timm에서 로드하기 위해 필요)
    model_name_map = {
        'tf_efficientnet_b4_ns': 'tf_efficientnet_b4_ns', 
        'convnext_base.fb_in22k': 'convnext_base.fb_in22k', 
        'tf_efficientnetv2_l.in21k_ft_in1k': 'tf_efficientnetv2_l.in21k_ft_in1k', 
        # 여기에 다른 모델 이름도 추가 가능 
    }
    
    all_logits = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
    
    # 추론에 사용할 공통 데이터로더 (V3의 384x384 설정 사용)
    test_dataset = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('test', cfg), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    for i, model_path in enumerate(all_model_paths):
        model_key = None
        for key in model_name_map:
            if key in model_path:
                model_key = key
                break
        
        if model_key is None:
            print(f"⚠️ 경고: 알 수 없는 모델 아키텍처: {model_path}. 건너뜁니다.")
            continue
            
        print(f"▶️ 모델 {i+1}/{len(all_model_paths)} 추론 중 (아키텍처: {model_key})")
        
        # 2. 모델 로드 및 추론
        current_model = timm.create_model(model_key, pretrained=False, num_classes=cfg.NUM_CLASSES)
        current_model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        current_model.to(cfg.DEVICE)
        current_model.eval()
        
        fold_logits = []
        for images in tqdm(test_loader, desc=f'Inference Model {i+1}'):
            images = images.to(cfg.DEVICE)
            outputs = current_model(images)
            fold_logits.append(outputs.cpu().numpy())

        all_logits += np.concatenate(fold_logits, axis=0)
        
    avg_logits = all_logits / len(all_model_paths)
    predictions = np.argmax(avg_logits, axis=1)

    return test_df['ID'], predictions


# ==============================================================================
# 4. 메인 실행 함수
# ==============================================================================

def main():
    cfg = Config()
    
    # 1. 데이터 준비
    download_and_extract_data(cfg.DATA_DIR)
    
    # 2. 실험 디렉토리 설정
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = f'./experiments/{cfg.MODEL_NAME}_{TIMESTAMP}'
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    print(f"🧪 실험 디렉토리 생성: {EXP_DIR}")
    
    # 3. 데이터 로드 및 가중치 계산
    train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    train_labels = train_df['target'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"⚖️ 클래스 가중치 계산 완료: {class_weights.numpy().round(3)}")
    
    # 4. K-Fold 학습 (V3 모델 학습)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['ID'], train_df['target'])):
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        best_f1, model_path = train_fold(fold, train_fold_df, val_fold_df, EXP_DIR, class_weights, cfg)
        
        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'model_path': model_path
        })
    
    # 학습 결과 요약
    results_df = pd.DataFrame(fold_results)
    print(f'\n{"="*50}')
    print(f'📊 V3 학습 결과 요약 - 모델: {cfg.MODEL_NAME}')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']].to_markdown(index=False))
    print(f'\n📌 CV 평균 F1: {results_df["f1"].mean():.4f}')
    
    # 5. 테스트 추론 및 최종 제출 파일 생성 (V1, V2, V3 통합 앙상블)
    test_ids, predictions = ultimate_inference_ensemble(cfg)
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'target': predictions
    })
    
    submission_filename = f'submission_{TIMESTAMP}_ULTIMATE_ENSEMBLE.csv'
    submission_path = os.path.join(EXP_DIR, submission_filename)
    submission.to_csv(submission_path, index=False)
    
    total_models = len(glob.glob(os.path.join(cfg.ENSEMBLE_MODEL_BASE_DIR, '*', 'best_model_fold_*.pth')))
    
    print('\n' + '='*70)
    print("✨ 최종 앙상블 제출 파일 생성 완료!")
    print(f"앙상블에 포함된 전체 모델 수: {total_models}개")
    print(f"최종 제출 파일 경로: {submission_path}")
    print('='*70)

if __name__ == '__main__':
    main()