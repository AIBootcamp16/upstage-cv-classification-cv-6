# main.py
# 문서 분류 - 로컬 실행용 (데이터 자동 다운로드 포함)

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
import random
import subprocess
import urllib.request

# ========== 데이터 다운로드 함수 ==========
def download_and_extract_data(data_dir='data'):
    """데이터 자동 다운로드 및 압축 해제"""
    data_path = Path(data_dir)
    
    if data_path.exists() and (data_path / 'train.csv').exists():
        print("✅ 데이터가 이미 존재합니다. 다운로드 건너뜀.\n")
        return
    
    print("="*70)
    print("📥 데이터 다운로드 중...")
    print("="*70)
    
    DATA_URL = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000372/data/data.tar.gz"
    DATA_FILE = "data.tar.gz"
    
    try:
        # 다운로드
        print(f"🌐 다운로드 시작: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("✅ 다운로드 완료")
        
        # 압축 해제
        print("\n📦 압축 해제 중...")
        import tarfile
        with tarfile.open(DATA_FILE, 'r:gz') as tar:
            tar.extractall('.')
        print("✅ 압축 해제 완료")
        
        # 압축 파일 삭제
        os.remove(DATA_FILE)
        print("🗑️  압축 파일 삭제 완료")
        
        # 데이터 구조 확인
        print("\n📁 데이터 구조:")
        if data_path.exists():
            for item in data_path.iterdir():
                print(f"  - {item.name}")
        
        print("\n✅ 데이터 준비 완료!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        print("수동으로 데이터를 다운로드하여 'data/' 폴더에 배치해주세요.")
        raise

# ========== Augraphy 체크 ==========
try:
    from augraphy import InkBleed, PaperFactory, DirtyDrum, Jpeg, Brightness, AugraphyPipeline
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("⚠️  Augraphy not installed. Using Albumentations only.")
    print("   Install with: pip install augraphy\n")

# ========== 설정 ==========
class Config:
    """학습 설정 클래스"""
    def __init__(self):
        # 데이터 경로 (로컬)
        self.DATA_DIR = 'data'
        self.TRAIN_DIR = 'data/train'
        self.TEST_DIR = 'data/test'
        
        # 모델 설정
        self.MODEL_NAME = 'tf_efficientnetv2_s'  # 또는 'tf_efficientnetv2_m', 'convnext_tiny'
        self.IMG_SIZE = 384
        self.NUM_CLASSES = 17
        
        # 학습 설정
        self.BATCH_SIZE = 8
        self.ACCUMULATION_STEPS = 2  # 효과적 배치 = 8 * 2 = 16
        self.EPOCHS = 15
        self.LR = 0.0001
        self.N_FOLDS = 5
        
        # 정규화 (과적합 방지)
        self.DROPOUT_RATE = 0.4
        self.WEIGHT_DECAY = 0.01
        self.LABEL_SMOOTHING = 0.05
        self.PATIENCE = 3
        
        # 증강 설정
        self.AUG_STRATEGY = 'hybrid'  # 'albumentations', 'augraphy', 'hybrid'
        self.AUGRAPHY_STRENGTH = 'light'  # 'light', 'medium', 'heavy'
        
        # 기타
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.2
        self.USE_CLASS_WEIGHTS = True
        self.SEED = 42
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    def print_config(self):
        """설정 출력"""
        print('='*70)
        print('⚙️  실험 설정')
        print('='*70)
        print(f'모델: {self.MODEL_NAME}')
        print(f'이미지 크기: {self.IMG_SIZE}')
        print(f'배치 크기: {self.BATCH_SIZE} (효과적: {self.BATCH_SIZE * self.ACCUMULATION_STEPS})')
        print(f'에폭: {self.EPOCHS}, 학습률: {self.LR}')
        print(f'Fold 수: {self.N_FOLDS}, Patience: {self.PATIENCE}')
        print(f'Dropout: {self.DROPOUT_RATE}, Weight Decay: {self.WEIGHT_DECAY}')
        print(f'증강 전략: {self.AUG_STRATEGY}')
        if self.AUG_STRATEGY in ['augraphy', 'hybrid']:
            print(f'Augraphy 강도: {self.AUGRAPHY_STRENGTH}')
        print(f'디바이스: {self.DEVICE}')
        print('='*70)

# 글로벌 설정
config = Config()
TIMESTAMP = None

# ========== Seed 설정 ==========
def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ========== 증강 함수들 ==========
def get_albumentations_train(image_size):
    """일반 이미지용 증강"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Affine(translate_percent=0.03, scale=(0.95, 1.05), rotate=(-3, 3), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_augraphy_train(image_size):
    """문서 특화 증강"""
    if not AUGRAPHY_AVAILABLE:
        return get_albumentations_train(image_size)
    
    ink_phase = [InkBleed(intensity_range=(0.1, 0.3), p=0.2)]
    paper_phase = [PaperFactory(p=0.2), DirtyDrum(p=0.1)]
    post_phase = [Jpeg(quality_range=(60, 95), p=0.2), Brightness(brightness_range=(0.95, 1.05), p=0.2)]
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    def apply_augraphy_safe(image, **kwargs):
        result = augraphy_pipeline.augment(image)["output"]
        if len(result.shape) == 2:
            result = np.stack([result] * 3, axis=-1)
        elif result.shape[-1] == 1:
            result = np.repeat(result, 3, axis=-1)
        elif result.shape[-1] == 4:
            result = result[:, :, :3]
        return result
    
    return A.Compose([
        A.Lambda(image=apply_augraphy_safe),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_hybrid_train(image_size, augraphy_strength='light'):
    """Augraphy + Albumentations 혼합"""
    if not AUGRAPHY_AVAILABLE:
        return get_albumentations_train(image_size)
    
    if augraphy_strength == 'light':
        ink_p, paper_p, post_p = 0.2, 0.2, 0.2
    elif augraphy_strength == 'medium':
        ink_p, paper_p, post_p = 0.4, 0.4, 0.3
    else:
        ink_p, paper_p, post_p = 0.6, 0.5, 0.4
    
    ink_phase = [InkBleed(intensity_range=(0.05, 0.15), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.5)]
    post_phase = [Jpeg(quality_range=(70, 95), p=post_p), Brightness(brightness_range=(0.95, 1.05), p=post_p)]
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    def apply_augraphy_safe(image, **kwargs):
        result = augraphy_pipeline.augment(image)["output"]
        if len(result.shape) == 2:
            result = np.stack([result] * 3, axis=-1)
        elif result.shape[-1] == 1:
            result = np.repeat(result, 3, axis=-1)
        elif result.shape[-1] == 4:
            result = result[:, :, :3]
        return result
    
    return A.Compose([
        A.Lambda(image=apply_augraphy_safe),
        A.Rotate(limit=3, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(p=0.2),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(image_size):
    """검증용 변환"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_train_transform(cfg):
    """Config 기반 증강 선택"""
    if cfg.AUG_STRATEGY == 'albumentations':
        return get_albumentations_train(cfg.IMG_SIZE)
    elif cfg.AUG_STRATEGY == 'augraphy':
        return get_augraphy_train(cfg.IMG_SIZE)
    elif cfg.AUG_STRATEGY == 'hybrid':
        return get_hybrid_train(cfg.IMG_SIZE, cfg.AUGRAPHY_STRENGTH)
    else:
        return get_albumentations_train(cfg.IMG_SIZE)

# ========== 데이터셋 ==========
class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['ID']
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
            return image
        else:
            label = row['label']
            return image, label

# ========== 학습 함수 ==========
def train_epoch(model, loader, criterion, optimizer, scheduler, cfg):
    model.train()
    losses = []
    optimizer.zero_grad()
    
    for idx, (images, labels) in enumerate(tqdm(loader, desc='Train')):
        images = images.to(cfg.DEVICE).contiguous()
        labels = labels.to(cfg.DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss = loss / cfg.ACCUMULATION_STEPS
        loss.backward()
        
        if (idx + 1) % cfg.ACCUMULATION_STEPS == 0 or (idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        losses.append(loss.item() * cfg.ACCUMULATION_STEPS)
    
    scheduler.step()
    return np.mean(losses)

def validate(model, loader, cfg):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images = images.to(cfg.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    f1 = f1_score(labels_list, preds_list, average='macro')
    return f1

# ========== 폴드 학습 ==========
def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    print(f'\n{"="*50}')
    print(f'Fold {fold} 학습 시작')
    print(f'{"="*50}')
    
    train_transform = get_train_transform(cfg)
    val_transform = get_val_transform(cfg.IMG_SIZE)
    
    train_dataset = DocumentDataset(train_df, cfg.TRAIN_DIR, train_transform)
    val_dataset = DocumentDataset(val_df, cfg.TRAIN_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES, drop_rate=cfg.DROPOUT_RATE)
    model = model.to(cfg.DEVICE)
    print(f'✅ 모델 로드: {cfg.MODEL_NAME}')
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    if cfg.USE_CLASS_WEIGHTS and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, cfg)
        val_f1 = validate(model, val_loader, cfg)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{cfg.EPOCHS} - Loss: {train_loss:.4f}, F1: {val_f1:.4f}, LR: {current_lr:.6f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'✅ Best F1: {best_f1:.4f}')
        else:
            patience_counter += 1
            print(f'⏳ Patience: {patience_counter}/{cfg.PATIENCE}')
        
        if patience_counter >= cfg.PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    model_filename = f'{exp_dir}/models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
    torch.save(best_model_state, model_filename)
    print(f'\nFold {fold} 완료 - Best F1: {best_f1:.4f}')
    
    return best_f1, model_filename

# ========== 앙상블 추론 ==========
def inference_ensemble(test_df, fold_info, cfg):
    print(f'\n{"="*50}')
    print(f'추론 시작 (모델 {len(fold_info)}개)')
    print(f'{"="*50}')
    
    test_transform = get_val_transform(cfg.IMG_SIZE)
    test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    models = []
    fold_f1s = []
    
    for fold, f1, model_path in fold_info:
        model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model = model.to(cfg.DEVICE)
        model.eval()
        models.append(model)
        fold_f1s.append(f1)
        print(f'✅ Fold {fold} (F1: {f1:.4f}) 로드')
    
    avg_f1 = np.mean(fold_f1s)
    weights = torch.tensor(fold_f1s, dtype=torch.float32)
    weights = weights / weights.sum()
    
    all_predictions = []
    
    for images in tqdm(test_loader, desc='Inference'):
        images = images.to(cfg.DEVICE)
        
        fold_preds = []
        for model in models:
            with torch.no_grad():
                pred = model(images)
            fold_preds.append(pred.cpu())
        
        fold_preds_tensor = torch.stack(fold_preds)
        weights_expanded = weights.unsqueeze(1).unsqueeze(2)
        ensemble_pred = (fold_preds_tensor * weights_expanded).sum(dim=0)
        final_class = ensemble_pred.argmax(dim=1).item()
        all_predictions.append(final_class)
    
    return all_predictions, avg_f1

# ========== 제출 파일 생성 ==========
def create_submission(test_df, predictions, avg_f1, exp_dir):
    filename = f'{exp_dir}/submission_{TIMESTAMP}_f1{avg_f1:.4f}.csv'
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    
    submission.to_csv(filename, index=False)
    
    print(f'\n{"="*50}')
    print(f'제출 파일: {filename}')
    print(f'{"="*50}')
    print(submission.head(10))
    print(f'\n예측 분포:')
    print(submission['target'].value_counts().sort_index())
    
    return filename

# ========== 메인 실행 ==========
if __name__ == '__main__':
    # 데이터 다운로드 (필요시)
    download_and_extract_data()
    
    # 초기화
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    set_seed(config.SEED)
    
    # 실험 폴더
    EXP_DIR = f'experiments/exp_{TIMESTAMP}'
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(f'{EXP_DIR}/models', exist_ok=True)
    
    print('\n'+'='*70)
    print('🚀 문서 분류 학습 시작')
    print('='*70)
    config.print_config()
    
    # 데이터 로드
    print(f'\n📂 데이터 로드 중...')
    train_df = pd.read_csv(f'{config.DATA_DIR}/train.csv')
    train_df['label'] = train_df['target']
    
    print(f'학습 데이터: {len(train_df)}장')
    print(f'클래스: {train_df["label"].nunique()}개')
    
    # 클래스 가중치
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    else:
        class_weights = None
    
    # K-Fold 학습
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), start=1):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        best_f1, model_path = train_fold(fold, train_fold_df, val_fold_df, EXP_DIR, class_weights, config)
        
        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'model_path': model_path
        })
    
    # 결과 출력
    results_df = pd.DataFrame(fold_results)
    print(f'\n{"="*50}')
    print('📊 학습 결과')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']])
    print(f'\n평균 F1: {results_df["f1"].mean():.4f}')
    print(f'최고 F1: {results_df["f1"].max():.4f}')
    
    # 결과 저장
    results_filename = f'{EXP_DIR}/fold_results_{TIMESTAMP}.csv'
    results_df.to_csv(results_filename, index=False)
    
    # 테스트 데이터
    if os.path.exists(f'{config.DATA_DIR}/test.csv'):
        test_df = pd.read_csv(f'{config.DATA_DIR}/test.csv')
    else:
        test_df = pd.read_csv(f'{config.DATA_DIR}/sample_submission.csv')
        test_df = test_df.drop('target', axis=1)
    
    print(f'\n테스트 데이터: {len(test_df)}장')
    
    # 앙상블 추론
    fold_info = [
        (row['fold'], row['f1'], row['model_path'])
        for _, row in results_df.iterrows()
    ]
    
    predictions, avg_f1 = inference_ensemble(test_df, fold_info, config)
    
    # 제출 파일 생성
    submission_filename = create_submission(test_df, predictions, avg_f1, EXP_DIR)
    
    print(f'\n{"="*70}')
    print('✅ 모든 작업 완료!')
    print(f'{"="*70}')
    print(f'📁 결과 파일: {results_filename}')
    print(f'📁 제출 파일: {submission_filename}')
    print(f'{"="*70}\n')