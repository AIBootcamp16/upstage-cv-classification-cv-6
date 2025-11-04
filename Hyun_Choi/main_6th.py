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
from torch.optim.lr_scheduler import LambdaLR
import math # Cosine Annealing을 위해 추가

# 경고 메시지 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. 설정 및 환경
# ==============================================================================

# Augraphy 라이브러리 체크 (필요시 설치)
try:
    from augraphy import InkBleed, PaperFactory, DirtyDrum, Jpeg, Brightness, AugraphyPipeline
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("⚠️ Augraphy 라이브러리를 찾을 수 없습니다. (pip install augraphy) - 문서 특화 증강 없이 진행됩니다.")

class Config:
    """최종 강건한 일반화를 위한 설정 (ConvNeXt + 극단적 노이즈 + TTA)"""
    PROJECT_NAME = "document-classification-ultimate"
    RUN_NAME = "ConvNeXt_ExtremeNoise_TTA_Warmup"
    
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k' 
    NUM_CLASSES = 17
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    USE_MIX_STRATEGY = True
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0 
    
    N_EPOCHS = 50 
    BATCH_SIZE = 16  
    IMAGE_SIZE = 384  
    
    LR = 1e-4 
    WARMUP_EPOCHS = 5
    
    GRADIENT_ACCUMULATION_STEPS = 2
    
    N_FOLDS = 5 
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.05
    
    SCHEDULER_T0 = 15 
    SCHEDULER_TMULT = 2
    PATIENCE = 10 
    
    TTA_SIZE = 7 
    
    DATA_DIR = 'data'
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

def get_augraphy_pipeline():
    """문서 이미지 특화 증강 파이프라인"""
    if not AUGRAPHY_AVAILABLE:
        return None

    ink_p, paper_p, post_p = 0.7, 0.6, 0.5 

    ink_phase = [InkBleed(intensity_range=(0.05, 0.20), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.7)]
    # JPEG 압축 품질을 낮춰 테스트 도메인 시뮬레이션
    post_phase = [Jpeg(quality_range=(30, 95), p=post_p), Brightness(brightness_range=(0.8, 1.2), p=post_p)] 

    return AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)


def get_transforms(stage, cfg):
    """극단적인 노이즈 증강(도메인 시뮬레이션) 및 TTA 통합"""
    augraphy_pipeline = get_augraphy_pipeline()
    
    if stage == 'train':
        albu_list = [
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), 
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8, p=0.7), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # 극단적인 노이즈 주입 및 Blur
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1), 
                A.Blur(blur_limit=5, p=1),
            ], p=0.8),
            A.CoarseDropout(max_holes=10, max_height=10, max_width=10, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        if AUGRAPHY_AVAILABLE and augraphy_pipeline is not None:
            albu_list.insert(1, A.Lambda(
                image=lambda x, **kwargs: ensure_3_channels(augraphy_pipeline.augment(x)['output']), 
                p=1.0)
            )
            
        return A.Compose(albu_list)
    
    # TTA를 위한 변환
    elif stage == 'test': 
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.JpegCompression(quality_lower=50, quality_upper=100, p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # Validation 및 기본 추론 (TTA가 아닌 경우)
    elif stage == 'val_base':
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return None

# ==============================================================================
# 3. 학습 및 추론 로직
# ==============================================================================

def mixup_cutmix_data(images, labels, alpha, mix_strategy='mixup'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    rand_index = torch.randperm(images.size(0)).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[rand_index, :]

    label_a, label_b = labels, labels[rand_index]
    return mixed_images, label_a, label_b, lam


def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    """단일 Fold 학습 (Warmup 스케줄러 적용) - Fold 번호는 1부터 시작"""
    print(f'\n{"="*50}')
    print(f'⚡️ Fold {fold} 학습 시작 - Model: {cfg.MODEL_NAME}')
    print(f'{"="*50}')
    
    run = None
    try:
        run = wandb.init(project=cfg.PROJECT_NAME, name=f"{cfg.RUN_NAME}_Fold_{fold}", config=vars(cfg), reinit=True)
    except Exception:
        print("WandB 초기화 실패. 로깅 없이 진행됩니다.")
    
    train_dataset = DocumentDataset(train_df, f'{cfg.DATA_DIR}/train', get_transforms('train', cfg))
    val_dataset = DocumentDataset(val_df, f'{cfg.DATA_DIR}/train', get_transforms('val_base', cfg))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES)
    model.to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE), 
                                    label_smoothing=cfg.LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # Warmup + CosineAnnealing 스케줄러 구현
    def lr_lambda(current_step):
        if current_step < cfg.WARMUP_EPOCHS * len(train_loader):
            # Warmup
            return float(current_step) / float(max(1, cfg.WARMUP_EPOCHS * len(train_loader)))
        # Cosine Annealing (Warmup 후 최대 LR에서 시작)
        T_total = cfg.N_EPOCHS * len(train_loader)
        T_rest = T_total - cfg.WARMUP_EPOCHS * len(train_loader)
        T_current = current_step - cfg.WARMUP_EPOCHS * len(train_loader)

        return 0.5 * (1. + math.cos(math.pi * T_current / T_rest))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    best_f1 = 0.0
    patience_counter = 0
    
    # 모델 저장 경로 설정 (Fold 번호 1부터 시작)
    model_path = os.path.join(exp_dir, f'best_model_fold_{fold}.pth')
    
    for epoch in range(cfg.N_EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds_list, train_labels_list = [], []
        
        optimizer.zero_grad() 
        
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f'Fold {fold} | Epoch {epoch+1}/{cfg.N_EPOCHS} (Train)')):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            # Mixup/CutMix 로직
            if cfg.USE_MIX_STRATEGY and random.random() < 0.8:
                strategy = 'mixup' if random.random() < 0.5 else 'cutmix'
                alpha = cfg.MIXUP_ALPHA if strategy == 'mixup' else cfg.CUTMIX_ALPHA
                
                mixed_images, label_a, label_b, lam = mixup_cutmix_data(images, labels, alpha, strategy)
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, label_a) + (1 - lam) * criterion(outputs, label_b)
                
                preds = outputs.argmax(dim=1) 
                target_labels = label_a
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                target_labels = labels
            
            # 경사 누적
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
            
            running_loss += loss.item() * images.size(0) * cfg.GRADIENT_ACCUMULATION_STEPS
            train_preds_list.extend(preds.cpu().numpy())
            train_labels_list.extend(target_labels.cpu().numpy())
        
        if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
             optimizer.step()
             scheduler.step()
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
                "Fold": fold, "Epoch": epoch, "LR": optimizer.param_groups[0]['lr'],
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
    """TTA를 통합한 이종 모델 앙상블 추론"""
    print('\n' + '='*70)
    print('🚀 최종 TTA 통합 앙상블 추론 시작')
    print('='*70)

    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    test_df = test_df.drop('target', axis=1, errors='ignore')
    
    all_model_paths = glob.glob(os.path.join(cfg.ENSEMBLE_MODEL_BASE_DIR, '*', 'best_model_fold_*.pth'))
    
    if not all_model_paths:
        raise FileNotFoundError(f"'{cfg.ENSEMBLE_MODEL_BASE_DIR}' 폴더에서 모델 파일이 발견되지 않았습니다. 모든 전략을 실행하고 이 코드를 재실행하세요.")
    
    print(f"앙상블에 사용할 총 모델 수: {len(all_model_paths)}개")
    
    model_name_map = {
        'tf_efficientnet_b4_ns': 'tf_efficientnet_b4_ns', 
        'convnext_base.fb_in22k': 'convnext_base.fb_in22k', 
        'tf_efficientnetv2_l.in21k_ft_in1k': 'tf_efficientnetv2_l.in21k_ft_in1k',
    }
    
    all_logits = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
    
    for i, model_path in enumerate(all_model_paths):
        model_key = None
        for key in model_name_map:
            if key in model_path:
                model_key = key
                break
        
        if model_key is None:
            continue
            
        current_model = timm.create_model(model_key, pretrained=False, num_classes=cfg.NUM_CLASSES)
        current_model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        current_model.to(cfg.DEVICE)
        current_model.eval()

        fold_logits_sum = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
        
        # 1. TTA를 적용하지 않은 기본 예측 (1회)
        test_dataset_base = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('val_base', cfg), is_test=True)
        test_loader_base = DataLoader(test_dataset_base, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        base_logits_list = []
        for images in test_loader_base:
            outputs = current_model(images.to(cfg.DEVICE))
            base_logits_list.append(outputs.cpu().numpy())
        fold_logits_sum += np.concatenate(base_logits_list, axis=0)
        
        # 2. TTA 적용 예측 (TTA_SIZE 회)
        for tta_iter in range(cfg.TTA_SIZE):
            test_dataset_tta = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('test', cfg), is_test=True)
            test_loader_tta = DataLoader(test_dataset_tta, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            tta_logits_list = []
            for images in test_loader_tta:
                outputs = current_model(images.to(cfg.DEVICE))
                tta_logits_list.append(outputs.cpu().numpy())
            
            fold_logits_sum += np.concatenate(tta_logits_list, axis=0)

        fold_avg_logits = fold_logits_sum / (cfg.TTA_SIZE + 1)
        all_logits += fold_avg_logits
        
    avg_logits = all_logits / len(all_model_paths)
    predictions = np.argmax(avg_logits, axis=1)

    return test_df['ID'], predictions


# ==============================================================================
# 4. 메인 실행 함수
# ==============================================================================

def main():
    cfg = Config()
    
    download_and_extract_data(cfg.DATA_DIR)
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = f'./experiments/{cfg.MODEL_NAME}_{TIMESTAMP}'
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    print(f"🧪 실험 디렉토리 생성: {EXP_DIR}")
    
    train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    train_labels = train_df['target'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"⚖️ 클래스 가중치 계산 완료: {class_weights.numpy().round(3)}")
    
    # K-Fold 학습 (V4 모델 학습)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df['ID'], train_df['target'])):
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # 🔥 Fold 번호를 1부터 시작하도록 +1
        fold_num = fold_idx + 1 
        
        best_f1, model_path = train_fold(fold_num, train_fold_df, val_fold_df, EXP_DIR, class_weights, cfg)
        
        fold_results.append({
            'fold': fold_num, # 수정된 fold_num 사용
            'f1': best_f1,
            'model_path': model_path
        })
    
    results_df = pd.DataFrame(fold_results)
    print(f'\n{"="*50}')
    print(f'📊 V4 학습 결과 요약 - 모델: {cfg.MODEL_NAME}')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']].to_markdown(index=False))
    print(f'\n📌 CV 평균 F1: {results_df["f1"].mean():.4f}')
    
    # 5. 테스트 추론 및 최종 제출 파일 생성 (V1, V2, V3, V4 통합 앙상블)
    test_ids, predictions = ultimate_inference_ensemble(cfg)
    
    submission = pd.DataFrame({'ID': test_ids, 'target': predictions})
    
    submission_filename = f'submission_{TIMESTAMP}_FINAL_ROBUST_ENSEMBLE.csv'
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