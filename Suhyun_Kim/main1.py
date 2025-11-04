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

# 경고 메시지 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. 설정 및 환경
# ==============================================================================

# Augraphy 라이브러리 체크
try:
    from augraphy import InkBleed, PaperFactory, DirtyDrum, Jpeg, Brightness, AugraphyPipeline
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("⚠️ Augraphy 라이브러리를 찾을 수 없습니다. (pip install augraphy) - 문서 특화 증강 없이 진행됩니다.")

class Config:
    """모델 및 학습에 필요한 하이퍼파라미터 설정 (Vision Transformer, TTA 적용)"""
    # 🔥 Vision Transformer 모델로 변경 (DINO 사전 학습 가중치)
    MODEL_NAME = 'vit_base_patch16_224.dino' 
    NUM_CLASSES = 17
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 하이퍼파라미터 
    N_EPOCHS = 30 
    # ViT는 224x224 해상도가 표준이며, BATCH_SIZE를 조금 더 늘려볼 수 있습니다.
    BATCH_SIZE = 16 
    # 🔥 ViT의 표준 입력 크기로 변경
    IMAGE_SIZE = 224  
    LR = 3e-5 # ViT는 낮은 LR이 안정적
    
    # 경사 누적 설정 (Effective Batch Size = 16 * 2 = 32 유지)
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # 정규화 및 최적화
    N_FOLDS = 5 
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.05
    
    # 스케줄러
    SCHEDULER_T0 = 10
    SCHEDULER_TMULT = 2
    PATIENCE = 7 
    
    # 🔥 TTA(Test Time Augmentation) 설정
    TTA_SIZE = 5 # TTA 횟수 (예: 원본, 좌우반전, 5가지 크롭/컬러 jitter 등)
    
    # 경로 및 증강
    DATA_DIR = 'data'
    AUG_STRATEGY = 'hybrid'


# ==============================================================================
# 2. 데이터 처리 함수 및 클래스
# ==============================================================================

def download_and_extract_data(data_dir='data'):
    """데이터 자동 다운로드 및 압축 해제"""
    data_path = Path(data_dir)
    
    if data_path.exists() and (data_path / 'train.csv').exists():
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
    def __init__(self, df, img_dir, transform=None, is_test=False, is_tta=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.is_tta = is_tta # TTA 시 True

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
            # TTA 추론 시, 이미지는 DataLoader에서 배치 단위로 처리
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

# 🔥 TTA 추론에 사용될 변환 함수
def get_tta_transforms(cfg):
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.SafeRotate(limit=10, p=1.0)
        ], p=0.8), # TTA 때 한 가지 변환만 강하게 적용
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_transforms(stage, cfg):
    """Albumentations 및 Augraphy를 통합한 이미지 변환 파이프라인"""
    augraphy_pipeline = get_augraphy_pipeline(cfg.AUG_STRATEGY)
    
    if stage == 'train':
        albu_list = [
            # 🔥 ViT 표준 입력 크기 224x224로 변경
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), 
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.CLAHE(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        if AUGRAPHY_AVAILABLE and augraphy_pipeline is not None:
            albu_list.insert(1, A.Lambda(
                image=lambda x, **kwargs: ensure_3_channels(augraphy_pipeline.augment(x)['output']), 
                p=1.0)
            )
            
        return A.Compose(albu_list)
    
    elif stage == 'val':
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # TTA를 위한 기본 변환 (Resize + Normalize)
    elif stage == 'test_base':
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return None

# ==============================================================================
# 3. 학습 및 추론 로직
# ==============================================================================

def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    """단일 Fold 학습 및 최적 모델 저장 (경사 누적 적용)"""
    print(f'\n{"="*50}')
    print(f'⚡️ Fold {fold} 학습 시작')
    print(f'{"="*50}')
    
    train_dataset = DocumentDataset(train_df, f'{cfg.DATA_DIR}/train', get_transforms('train', cfg))
    val_dataset = DocumentDataset(val_df, f'{cfg.DATA_DIR}/train', get_transforms('val', cfg))
    
    # BATCH_SIZE=16 사용
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 🔥 ViT 모델 로드
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step(epoch + (step + 1) / len(train_loader)) 
                optimizer.zero_grad() 
            
            running_loss += loss.item() * images.size(0) * cfg.GRADIENT_ACCUMULATION_STEPS
            train_preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
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
                
    return best_f1, model_path

@torch.no_grad()
def inference_ensemble(exp_dir, cfg):
    """5-Fold 학습 모델들을 사용한 Logit 평균 앙상블 추론 (TTA 적용)"""
    print('\n' + '='*50)
    print(f'🚀 5-Fold 앙상블 추론 시작 (TTA: {cfg.TTA_SIZE}회 적용)')
    print('='*50)

    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    test_df = test_df.drop('target', axis=1, errors='ignore')
    
    # TTA를 적용하지 않는 기본 추론 DataLoader
    test_dataset_base = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', get_transforms('test_base', cfg), is_test=True)
    test_loader_base = DataLoader(test_dataset_base, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # TTA를 위한 변환 함수
    tta_transforms = get_tta_transforms(cfg)

    model_paths = sorted(glob.glob(os.path.join(exp_dir, 'best_model_fold_*.pth')))
    if not model_paths:
        raise FileNotFoundError(f"모델 파일이 '{exp_dir}'에서 발견되지 않았습니다. 학습을 먼저 진행하세요.")
        
    print(f"앙상블에 사용할 모델 수: {len(model_paths)}개")

    all_logits = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
    
    for i, model_path in enumerate(model_paths):
        print(f"▶️ 모델 {i+1}/{len(model_paths)} 추론 중...")
        
        model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.to(cfg.DEVICE)
        model.eval()
        
        fold_logits_sum = np.zeros((len(test_df), cfg.NUM_CLASSES), dtype=np.float32)
        
        # 1. TTA를 적용하지 않은 기본 예측 (1회)
        base_logits_list = []
        for images in tqdm(test_loader_base, desc=f'Inference Model {i+1} (Base)'):
            images = images.to(cfg.DEVICE)
            outputs = model(images)
            base_logits_list.append(outputs.cpu().numpy())
        fold_logits_sum += np.concatenate(base_logits_list, axis=0)
        
        # 2. 🔥 TTA 적용 예측 (cfg.TTA_SIZE 회)
        for tta_iter in range(cfg.TTA_SIZE):
            # TTA를 위한 임시 데이터셋/로더 (TTA 변환은 런타임에 적용)
            test_dataset_tta = DocumentDataset(test_df, f'{cfg.DATA_DIR}/test', tta_transforms, is_test=True, is_tta=True)
            test_loader_tta = DataLoader(test_dataset_tta, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            tta_logits_list = []
            for images in tqdm(test_loader_tta, desc=f'Inference Model {i+1} (TTA {tta_iter+1}/{cfg.TTA_SIZE})'):
                images = images.to(cfg.DEVICE)
                outputs = model(images)
                tta_logits_list.append(outputs.cpu().numpy())
            
            fold_logits_sum += np.concatenate(tta_logits_list, axis=0)

        # Logit 평균: (기본 예측 1회 + TTA 예측 TTA_SIZE회)
        fold_avg_logits = fold_logits_sum / (cfg.TTA_SIZE + 1)
        all_logits += fold_avg_logits
        
    avg_logits = all_logits / len(model_paths)
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
    # 🔥 실험 디렉토리 이름에 TTA 플래그 추가
    EXP_DIR = f'./experiments/{cfg.MODEL_NAME}_tta{cfg.TTA_SIZE}_{TIMESTAMP}'
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    print(f"🧪 실험 디렉토리 생성: {EXP_DIR}")
    
    # 3. 데이터 로드 및 가중치 계산
    train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    train_labels = train_df['target'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"⚖️ 클래스 가중치 계산 완료: {class_weights.numpy().round(3)}")
    
    # 4. K-Fold 학습
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
    print('📊 최종 학습 결과 요약')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']].to_markdown(index=False))
    print(f'\n📌 CV 평균 F1: {results_df["f1"].mean():.4f}')
    
    # 5. 테스트 추론 및 제출 파일 생성
    # 이 함수는 TTA가 적용되어 추론 시간이 길어질 수 있습니다.
    test_ids, predictions = inference_ensemble(EXP_DIR, cfg) 
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'target': predictions
    })
    
    avg_f1 = results_df["f1"].mean()
    submission_filename = f'submission_{TIMESTAMP}_ensemble_tta{cfg.TTA_SIZE}_avgf1{avg_f1:.4f}.csv'
    submission_path = os.path.join(EXP_DIR, submission_filename)
    submission.to_csv(submission_path, index=False)
    
    print('\n' + '='*50)
    print("✨ 모든 프로세스 완료!")
    print(f"최종 제출 파일 경로: {submission_path}")
    print('='*50)

if __name__ == '__main__':
    main()