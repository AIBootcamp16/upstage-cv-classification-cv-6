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
import math 
import copy 

# 경고 메시지 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 0. EMA (Exponential Moving Average) 모델 정의
# ==============================================================================
class EMAModel:
    """EMA (지수 이동 평균) 가중치 업데이트를 위한 헬퍼 클래스"""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.steps = 0

    def update(self):
        """매 Step마다 이동 평균 가중치 업데이트"""
        self.steps += 1
        # Cosine EMA Decay 적용
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps)) 
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        
    def apply_shadow(self, save_path=None):
        """저장 또는 추론을 위해 모델에 EMA 가중치 적용"""
        original_state_dict = self.model.state_dict()
        self.model.load_state_dict(self.shadow)
        
        if save_path:
            torch.save(self.model.state_dict(), save_path)
        
        self.model.load_state_dict(original_state_dict)


# ==============================================================================
# 1. 설정 및 환경 (시간 단축 및 일반화 개선)
# ==============================================================================
# 🚨 Augraphy 제거 (일반화 실패의 주요 원인으로 판단)
AUGRAPHY_AVAILABLE = False
# print("⚠️ Augraphy 라이브러리 사용을 중지합니다. (일반화 개선 목적)")

class Config:
    """최종 우승을 위한 설정 (ConvNeXt + EMA + TTA + PL Gap Fix)"""
    PROJECT_NAME = "document-classification-v7"
    RUN_NAME = "ConvNeXt_EMA_PL_FIX_Refined_V9"
    
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k' 
    NUM_CLASSES = 17
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mixup/CutMix 확률 감소 (과적합 방지)
    USE_MIX_STRATEGY = True
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0 
    MIX_PROB = 0.5 # 🔥 기존 0.8 -> 0.5로 감소
    
    # 🔥 EMA Decay 조정 (가중치 변화 속도 약간 증가 -> 덜 공격적인 정규화)
    EMA_DECAY = 0.999 
    
    # 🔥 실험 시간 단축: Epoch 감소
    N_EPOCHS = 30 # 기존 50 -> 30으로 감소
    
    # 🔥 실험 시간 단축 및 OOM 완화: BATCH_SIZE 조정
    BATCH_SIZE = 16  # 기존 OOM 픽스 8 -> 16으로 증가 (시스템 리소스 확인 후)
    IMAGE_SIZE = 384  
    
    # 🔥 실험 시간 단축: Accumulation Steps 조정 (Effective Batch Size 32 유지)
    GRADIENT_ACCUMULATION_STEPS = 2 # 기존 OOM 픽스 4 -> 2로 감소 (BATCH_SIZE 16에 맞춤)
    
    # 최적화
    LR = 1e-4 
    WARMUP_EPOCHS = 3 # Warmup도 5->3으로 약간 감소
    
    # K-Fold 및 정지
    N_FOLDS = 5 
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.05
    PATIENCE = 5 # 🔥 Early Stopping 민감도 증가 (오버피팅 방지)
    
    # 🔥 추론 시간 단축: TTA 크기 감소
    TTA_SIZE = 3 # 기존 7 -> 3으로 감소
    
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

def get_transforms(stage, cfg):
    """일반화 개선을 위해 Augraphy를 제거하고 표준 Augmentation만 사용"""
    
    if stage == 'train':
        # 🔥 Augraphy 제거. 기본적인 강력한 Augmentation만 유지.
        albu_list = [
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), 
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8, p=0.7), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.GaussNoise(var_limit=(5.0, 30.0), p=1), # 노이즈 범위 완화
                A.Blur(blur_limit=3, p=1), # 블러 범위 완화
            ], p=0.6), # OneOf 확률 감소
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3), # Dropout 강도 완화
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
            
        return A.Compose(albu_list)
    
    # 🔥 추론 시간 단축: TTA를 위한 변환 (단순화)
    elif stage == 'test': 
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
                A.JpegCompression(quality_lower=70, quality_upper=100, p=1.0), # JPEG 노이즈 완화
            ], p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # Validation 및 기본 추론 
    elif stage == 'val_base':
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return None

def mixup_cutmix_data(images, labels, alpha, mix_strategy='mixup'):
    """Mixup / CutMix 구현"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    rand_index = torch.randperm(images.size(0)).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[rand_index, :]

    label_a, label_b = labels, labels[rand_index]
    return mixed_images, label_a, label_b, lam


# ==============================================================================
# 3. 학습 및 추론 로직 (EMA 통합 및 시간 단축)
# ==============================================================================

def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    """단일 Fold 학습 (시간 단축 및 일반화 개선)"""
    print(f'\n{"="*50}')
    print(f'⚡️ Fold {fold} 학습 시작 - Model: {cfg.MODEL_NAME}')
    print(f'   (Batch Size: {cfg.BATCH_SIZE}, Accumulation: {cfg.GRADIENT_ACCUMULATION_STEPS}, Effective: {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS})')
    print(f'   (Epochs: {cfg.N_EPOCHS}, Patience: {cfg.PATIENCE})')
    print(f'{"="*50}')
    
    run = None
    try:
        run = wandb.init(project=cfg.PROJECT_NAME, name=f"{cfg.RUN_NAME}_Fold_{fold}", config=vars(cfg), reinit=True)
    except Exception:
        print("WandB 초기화 실패. 로깅 없이 진행됩니다.")
    
    train_dataset = DocumentDataset(train_df, f'{cfg.DATA_DIR}/train', get_transforms('train', cfg))
    val_dataset = DocumentDataset(val_df, f'{cfg.DATA_DIR}/train', get_transforms('val_base', cfg))
    
    # 🔥 BATCH_SIZE 16
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES)
    model.to(cfg.DEVICE)
    
    ema_model = EMAModel(model, cfg.EMA_DECAY)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE), 
                                    label_smoothing=cfg.LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # Warmup + CosineAnnealing 스케줄러 구현
    def lr_lambda(current_step):
        if current_step < cfg.WARMUP_EPOCHS * len(train_loader):
            return float(current_step) / float(max(1, cfg.WARMUP_EPOCHS * len(train_loader)))
        T_total = cfg.N_EPOCHS * len(train_loader)
        T_rest = T_total - cfg.WARMUP_EPOCHS * len(train_loader)
        T_current = current_step - cfg.WARMUP_EPOCHS * len(train_loader)
        return 0.5 * (1. + math.cos(math.pi * T_current / T_rest))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
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
            
            # 🔥 Mixup/CutMix 확률 50% 적용 (과적합 방지)
            if cfg.USE_MIX_STRATEGY and random.random() < cfg.MIX_PROB:
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
            
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                ema_model.update()
            
            running_loss += loss.item() * images.size(0) * cfg.GRADIENT_ACCUMULATION_STEPS
            train_preds_list.extend(preds.cpu().numpy())
            train_labels_list.extend(target_labels.cpu().numpy())
        
        if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
             optimizer.step()
             scheduler.step()
             optimizer.zero_grad()
             ema_model.update()

        epoch_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels_list, train_preds_list, average='macro')
        
        # Validation (EMA 적용)
        model.eval()
        val_preds_list, val_labels_list = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Fold {fold} | Epoch {epoch+1}/{cfg.N_EPOCHS} (Val)'):
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                
                ema_model.apply_shadow() 
                outputs = model(images)
                ema_model.apply_shadow(save_path=None) 
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels_list, val_preds_list, average='macro')
        
        print(f"  [Result] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} (Best: {best_f1:.4f})")
        
        if run:
            run.log({
                "Fold": fold, "Epoch": epoch, "LR": optimizer.param_groups[0]['lr'],
                "Train/Loss": epoch_loss, "Train/F1": train_f1,
                "Val/Loss": val_loss, "Val/F1": val_f1
            })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            ema_model.apply_shadow(save_path=model_path)
            print(f"  🏆 New Best F1: {best_f1:.4f}. EMA Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE: # 🔥 PATIENCE 5
                print(f"  🛑 Early stopping on Fold {fold} at Epoch {epoch+1}")
                break
                
    if run:
        run.finish()
    return best_f1, model_path

@torch.no_grad()
def ultimate_inference_ensemble(cfg):
    """TTA를 통합한 이종 모델 앙상블 추론 (TTA_SIZE=3)"""
    print('\n' + '='*70)
    print(f'🚀 최종 TTA 통합 앙상블 추론 시작 (TTA Size: {cfg.TTA_SIZE})')
    print('='*70)

    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    test_df = test_df.drop('target', axis=1, errors='ignore')
    
    all_model_paths = glob.glob(os.path.join(cfg.ENSEMBLE_MODEL_BASE_DIR, '*', 'best_model_fold_*.pth'))
    
    if not all_model_paths:
        print("⚠️ 경고: 앙상블 폴더에서 모델이 발견되지 않았습니다. 현재 학습된 5개 모델만 사용합니다.")
        # 이 코드 실행 직후 저장된 모델만 사용하도록 제한 (현재 디렉토리 기준)
        all_model_paths = glob.glob(os.path.join(os.path.abspath('.'), '*', 'best_model_fold_*.pth'))
        if not all_model_paths:
            raise FileNotFoundError(f"모델 파일이 발견되지 않았습니다. 학습을 먼저 완료해야 합니다.")

    print(f"앙상블에 사용할 총 모델 수: {len(all_model_paths)}개")
    
    model_name_map = {
        'tf_efficientnet_b4_ns': 'tf_efficientnet_b4_ns', 
        'convnext_base.fb_in22k': 'convnext_base.fb_in22k_ft_in1k',
        'tf_efficientnetv2_l.in21k_ft_in1k': 'tf_efficientnetv2_l.in21k_ft_in1k',
    }
    
    all_logits = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
    
    for i, model_path in enumerate(tqdm(all_model_paths, desc="앙상블 추론 진행")):
        model_key = None
        for key in model_name_map:
            if key in model_path:
                model_key = key
                break
        
        if model_key is None:
            # 이전에 학습된 모델 파일 경로의 모델 이름을 찾아 Config.MODEL_NAME으로 대체
            current_model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)
        else:
            current_model = timm.create_model(model_name_map[model_key], pretrained=False, num_classes=cfg.NUM_CLASSES)
            
        current_model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        current_model.to(cfg.DEVICE)
        current_model.eval()

        fold_logits_sum = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
        
        # 기본 예측 (1회)
        test_dataset_base = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('val_base', cfg), is_test=True)
        test_loader_base = DataLoader(test_dataset_base, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        base_logits_list = []
        for images in test_loader_base:
            outputs = current_model(images.to(cfg.DEVICE))
            base_logits_list.append(outputs.cpu().numpy())
        fold_logits_sum += np.concatenate(base_logits_list, axis=0)
        
        # TTA 적용 예측 (TTA_SIZE 회)
        for tta_iter in range(cfg.TTA_SIZE): # 🔥 TTA_SIZE 3
            test_dataset_tta = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('test', cfg), is_test=True)
            test_loader_tta = DataLoader(test_dataset_tta, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            tta_logits_list = []
            for images in test_loader_tta:
                outputs = current_model(images.to(cfg.DEVICE))
                tta_logits_list.append(outputs.cpu().numpy())
            
            fold_logits_sum += np.concatenate(tta_logits_list, axis=0)

        # Logit 평균: (기본 예측 1회 + TTA 예측 TTA_SIZE회)
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
    
    # K-Fold 학습 
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df['ID'], train_df['target'])):
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        fold_num = fold_idx + 1 
        
        best_f1, model_path = train_fold(fold_num, train_fold_df, val_fold_df, EXP_DIR, class_weights, cfg)
        
        fold_results.append({
            'fold': fold_num, 
            'f1': best_f1,
            'model_path': model_path
        })
    
    results_df = pd.DataFrame(fold_results)
    print(f'\n{"="*50}')
    print(f'📊 V9 학습 결과 요약 - 모델: {cfg.MODEL_NAME}')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']].to_markdown(index=False))
    print(f'\n📌 CV 평균 F1: {results_df["f1"].mean():.4f}')
    
    # 5. 테스트 추론 및 최종 제출 파일 생성 (앙상블)
    test_ids, predictions = ultimate_inference_ensemble(cfg)
    
    submission = pd.DataFrame({'ID': test_ids, 'target': predictions})
    
    submission_filename = f'submission_{TIMESTAMP}_PL_FIX_V9_CV{results_df["f1"].mean():.4f}.csv'
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