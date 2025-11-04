# main1.py
# 문서 분류 - 실험용 스크립트 (증강/정규화 튠 포함)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
import math
import urllib.request


# ==============================================================================
# 1. 데이터 다운로드
# ==============================================================================


def download_and_extract_data(data_dir='data'):
    """데이터 자동 다운로드 및 압축 해제"""
    data_path = Path(data_dir)

    if data_path.exists() and (data_path / 'train.csv').exists():
        print("✅ 데이터가 이미 존재합니다. 다운로드 건너뜀.\n")
        return

    print("=" * 70)
    print("📥 데이터 다운로드 중...")
    print("=" * 70)

    DATA_URL = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000372/data/data.tar.gz"
    DATA_FILE = "data.tar.gz"

    try:
        print(f"🌐 다운로드 시작: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("✅ 다운로드 완료")

        print("\n📦 압축 해제 중...")
        import tarfile

        with tarfile.open(DATA_FILE, 'r:gz') as tar:
            tar.extractall('.')
        print("✅ 압축 해제 완료")

        os.remove(DATA_FILE)
        print("🗑️  압축 파일 삭제 완료")

        if data_path.exists():
            print("\n📁 데이터 구조:")
            for item in data_path.iterdir():
                print(f"  - {item.name}")

        print("\n✅ 데이터 준비 완료!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        print("수동으로 데이터를 다운로드하여 'data/' 폴더에 배치해주세요.")
        raise


# ==============================================================================
# 2. Augraphy 체크
# ==============================================================================


try:
    from augraphy import (
        InkBleed,
        PaperFactory,
        DirtyDrum,
        Jpeg,
        Brightness,
        AugraphyPipeline,
    )

    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("⚠️  Augraphy not installed. Using Albumentations only.")
    print("   Install with: pip install augraphy\n")


# ==============================================================================
# 3. 설정
# ==============================================================================


class Config:
    """학습 설정 클래스"""

    def __init__(self):
        self.DATA_DIR = 'data'
        self.TRAIN_DIR = 'data/train'
        self.TEST_DIR = 'data/test'

        self.MODEL_NAME = 'tf_efficientnetv2_m'
        self.IMG_SIZE = 384
        self.NUM_CLASSES = 17

        self.BATCH_SIZE = 12
        self.VAL_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 32
        self.TRAIN_WORKERS = 4
        self.EVAL_WORKERS = 4
        self.ACCUMULATION_STEPS = 4
        self.EPOCHS = 20
        self.LR = 3e-4
        self.N_FOLDS = 5

        self.DROPOUT_RATE = 0.5
        self.WEIGHT_DECAY = 0.05
        self.LABEL_SMOOTHING = 0.1
        self.PATIENCE = 5

        # 증강 & 정규화 옵션
        self.AUG_STRATEGY = 'hybrid'
        self.AUGRAPHY_STRENGTH = 'medium'
        self.AUG_PERSPECTIVE = True
        self.AUG_PERSPECTIVE_SCALE = (0.03, 0.07)
        self.AUG_GRID_DISTORTION = True
        self.AUG_GRID_NUM_STEPS = 5
        self.AUG_COARSE_DROPOUT = True
        self.AUG_COARSE_PARAMS = dict(max_holes=6, max_height=32, max_width=32, fill_value=255)
        self.AUG_COLOR_JITTER = True

        # Mixup / CutMix
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.4
        self.MIXUP_PROB = 0.5
        self.USE_CUTMIX = True
        self.CUTMIX_ALPHA = 1.0
        self.CUTMIX_PROB = 0.5

        self.MAX_GRAD_NORM = 1.0
        self.SCHEDULER = 'cosine'
        self.SCHEDULER_STEP = 'epoch'
        self.SCHEDULER_T0 = 10
        self.SCHEDULER_TMULT = 2
        self.ETA_MIN = 1e-6
        self.USE_CLASS_WEIGHTS = True
        self.SEED = 42
        self.DEVICE = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.USE_AMP = torch.cuda.is_available()
        self.TTA_HORIZONTAL_FLIP = True

    def print_config(self):
        print('=' * 70)
        print('⚙️  실험 설정')
        print('=' * 70)
        print(f'모델: {self.MODEL_NAME}')
        print(f'이미지 크기: {self.IMG_SIZE}')
        print(
            f'배치 크기(Train/Val/Test): '
            f'{self.BATCH_SIZE}/{self.VAL_BATCH_SIZE}/{self.TEST_BATCH_SIZE} '
            f'(효과적 Train: {self.BATCH_SIZE * self.ACCUMULATION_STEPS})'
        )
        print(f'에폭: {self.EPOCHS}, 학습률: {self.LR}')
        print(f'Fold 수: {self.N_FOLDS}, Patience: {self.PATIENCE}')
        print(f'Dropout: {self.DROPOUT_RATE}, Weight Decay: {self.WEIGHT_DECAY}')
        print(f'Scheduler: {self.SCHEDULER} (step: {self.SCHEDULER_STEP})')
        print(
            f'Mixup: {self.USE_MIXUP} (alpha={self.MIXUP_ALPHA}, prob={self.MIXUP_PROB}) | '
            f'CutMix: {self.USE_CUTMIX} (alpha={self.CUTMIX_ALPHA}, prob={self.CUTMIX_PROB})'
        )
        if self.AUG_STRATEGY in ['augraphy', 'hybrid']:
            print(f'Augraphy 강도: {self.AUGRAPHY_STRENGTH}')
        print(
            f'추가 증강 - Perspective: {self.AUG_PERSPECTIVE}, '
            f'GridDistortion: {self.AUG_GRID_DISTORTION}, '
            f'CoarseDropout: {self.AUG_COARSE_DROPOUT}'
        )
        print(f'AMP 사용: {self.USE_AMP} | 디바이스: {self.DEVICE}')
        print(f'Max Grad Norm: {self.MAX_GRAD_NORM}')
        print('=' * 70)


config = Config()
TIMESTAMP = None


# ==============================================================================
# 4. Seed 설정
# ==============================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================================================================
# 5. 증강 함수
# ==============================================================================


def get_albumentations_train(cfg):
    transforms = [
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Affine(translate_percent=0.04, scale=(0.94, 1.06), shear=(-3, 3), rotate=(-5, 5), p=0.4),
    ]

    if cfg.AUG_PERSPECTIVE:
        transforms.append(A.Perspective(scale=cfg.AUG_PERSPECTIVE_SCALE, p=0.25))

    if cfg.AUG_GRID_DISTORTION:
        transforms.append(
            A.GridDistortion(num_steps=cfg.AUG_GRID_NUM_STEPS, distort_limit=0.1, p=0.25)
        )

    transforms.extend(
        [
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.4)
            if cfg.AUG_COLOR_JITTER
            else A.NoOp(),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ]
    )

    if cfg.AUG_COARSE_DROPOUT:
        transforms.append(
            A.CoarseDropout(
                max_holes=cfg.AUG_COARSE_PARAMS['max_holes'],
                max_height=cfg.AUG_COARSE_PARAMS['max_height'],
                max_width=cfg.AUG_COARSE_PARAMS['max_width'],
                fill_value=cfg.AUG_COARSE_PARAMS['fill_value'],
                p=0.25,
            )
        )

    transforms.extend(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return A.Compose([t for t in transforms if not isinstance(t, A.NoOp)])


def get_augraphy_train(cfg):
    if not AUGRAPHY_AVAILABLE:
        return get_albumentations_train(cfg)

    strength = cfg.AUGRAPHY_STRENGTH
    if strength == 'light':
        ink_p, paper_p, post_p = 0.2, 0.2, 0.15
    elif strength == 'medium':
        ink_p, paper_p, post_p = 0.35, 0.35, 0.25
    else:
        ink_p, paper_p, post_p = 0.5, 0.45, 0.35

    ink_phase = [InkBleed(intensity_range=(0.05, 0.18), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.5)]
    post_phase = [
        Jpeg(quality_range=(65, 95), p=post_p),
        Brightness(brightness_range=(0.9, 1.1), p=post_p),
    ]
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

    def apply_augraphy_safe(image, **kwargs):
        result = augraphy_pipeline.augment(image)["output"]
        if result.ndim == 2:
            result = np.stack([result] * 3, axis=-1)
        elif result.shape[-1] == 1:
            result = np.repeat(result, 3, axis=-1)
        elif result.shape[-1] == 4:
            result = result[:, :, :3]
        return result

    return A.Compose(
        [
            A.Lambda(image=apply_augraphy_safe),
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_hybrid_train(cfg):
    if not AUGRAPHY_AVAILABLE:
        return get_albumentations_train(cfg)

    augraphy_part = get_augraphy_train(cfg)
    albumentations_part = get_albumentations_train(cfg)

    def pipeline(image, **kwargs):
        if random.random() < 0.5:
            return {'image': augraphy_part(image=image)['image']}
        return {'image': albumentations_part(image=image)['image']}

    return A.Compose([A.Lambda(image=lambda img, **_: pipeline(img)['image'])])


def get_val_transform(cfg):
    return A.Compose(
        [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_train_transform(cfg):
    if cfg.AUG_STRATEGY == 'albumentations':
        return get_albumentations_train(cfg)
    if cfg.AUG_STRATEGY == 'augraphy':
        return get_augraphy_train(cfg)
    if cfg.AUG_STRATEGY == 'hybrid':
        return get_hybrid_train(cfg)
    return get_albumentations_train(cfg)


# ==============================================================================
# 6. 학습 유틸 (Mixup / CutMix)
# ==============================================================================


def mixup_data(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    targets_a = labels
    targets_b = labels[perm]
    return mixed_images, targets_a, targets_b, lam


def cutmix_data(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = images.size()
    perm = torch.randperm(batch_size, device=images.device)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    targets_a = labels
    targets_b = labels[perm]
    return images, targets_a, targets_b, lam


def apply_mix_strategy(images, labels, cfg):
    use_mixup = cfg.USE_MIXUP and random.random() < cfg.MIXUP_PROB
    use_cutmix = cfg.USE_CUTMIX and random.random() < cfg.CUTMIX_PROB

    if use_cutmix:
        return cutmix_data(images, labels, cfg.CUTMIX_ALPHA), 'cutmix'
    if use_mixup:
        return mixup_data(images, labels, cfg.MIXUP_ALPHA), 'mixup'
    return (images, labels, labels, 1.0), 'none'


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.SCHEDULER_T0,
            T_mult=cfg.SCHEDULER_TMULT,
            eta_min=cfg.ETA_MIN,
        )
        return scheduler, 'epoch'
    if cfg.SCHEDULER == 'onecycle':
        effective_steps_per_epoch = math.ceil(steps_per_epoch / cfg.ACCUMULATION_STEPS)
        total_steps = max(1, cfg.EPOCHS * effective_steps_per_epoch)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.LR,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )
        return scheduler, 'batch'
    if cfg.SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=cfg.ETA_MIN,
            verbose=True,
        )
        return scheduler, 'metric'
    return None, None


# ==============================================================================
# 7. Dataset
# ==============================================================================


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

        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label


# ==============================================================================
# 8. 학습 및 검증 루프
# ==============================================================================


def train_epoch(model, loader, criterion, optimizer, scheduler, scheduler_step_mode, scaler, cfg, epoch):
    model.train()
    losses = []
    optimizer.zero_grad()

    use_amp = cfg.USE_AMP and 'cuda' in cfg.DEVICE

    for idx, (images, labels) in enumerate(tqdm(loader, desc=f'Train | Epoch {epoch + 1}')):
        images = images.to(cfg.DEVICE, non_blocking=True).contiguous()
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        (images, targets_a, targets_b, lam), mix_type = apply_mix_strategy(images, labels, cfg)
        images = images.contiguous()

        with autocast(enabled=use_amp):
            outputs = model(images)
            if mix_type == 'mixup' or mix_type == 'cutmix':
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)

        loss = loss / cfg.ACCUMULATION_STEPS
        losses.append(loss.detach().item() * cfg.ACCUMULATION_STEPS)

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % cfg.ACCUMULATION_STEPS == 0 or (idx + 1) == len(loader):
            if cfg.MAX_GRAD_NORM is not None:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.MAX_GRAD_NORM)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None and scheduler_step_mode == 'batch':
                scheduler.step()

    return float(np.mean(losses))


def validate(model, loader, cfg):
    model.eval()
    preds_list = []
    labels_list = []
    use_amp = cfg.USE_AMP and 'cuda' in cfg.DEVICE

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images = images.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(images)
            preds = outputs.argmax(dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return f1_score(labels_list, preds_list, average='macro')


# ==============================================================================
# 9. Fold 학습
# ==============================================================================


def train_fold(fold, train_df, val_df, exp_dir, class_weights, cfg):
    print(f"\n{'=' * 50}")
    print(f'Fold {fold} 학습 시작')
    print(f"{'=' * 50}")

    train_dataset = DocumentDataset(train_df, cfg.TRAIN_DIR, get_train_transform(cfg))
    val_dataset = DocumentDataset(val_df, cfg.TRAIN_DIR, get_val_transform(cfg))

    pin_memory = 'cuda' in cfg.DEVICE
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=cfg.TRAIN_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.EVAL_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=cfg.EVAL_WORKERS > 0,
    )

    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES, drop_rate=cfg.DROPOUT_RATE)
    model = model.to(cfg.DEVICE)
    print(f'✅ 모델 로드: {cfg.MODEL_NAME}')

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler(enabled=cfg.USE_AMP and 'cuda' in cfg.DEVICE)
    scheduler, scheduler_step_mode = build_scheduler(optimizer, cfg, len(train_loader))

    if cfg.USE_CLASS_WEIGHTS and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE), label_smoothing=cfg.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scheduler_step_mode, scaler, cfg, epoch)
        val_f1 = validate(model, val_loader, cfg)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{cfg.EPOCHS} - Loss: {train_loss:.4f}, F1: {val_f1:.4f}, LR: {current_lr:.6f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'✅ Best F1: {best_f1:.4f}')
        else:
            patience_counter += 1
            print(f'⏳ Patience: {patience_counter}/{cfg.PATIENCE}')

        if patience_counter >= cfg.PATIENCE:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if scheduler is not None:
            if scheduler_step_mode == 'epoch':
                if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()
            elif scheduler_step_mode == 'metric':
                scheduler.step(val_f1)

    if best_model_state is None:
        best_model_state = model.state_dict()

    model_filename = f'{exp_dir}/models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
    torch.save(best_model_state, model_filename)
    print(f'\nFold {fold} 완료 - Best F1: {best_f1:.4f}')

    return best_f1, model_filename


# ==============================================================================
# 10. 앙상블 추론
# ==============================================================================


def inference_ensemble(test_df, fold_info, cfg):
    print(f"\n{'=' * 50}")
    print(f'추론 시작 (모델 {len(fold_info)}개)')
    print(f"{'=' * 50}")

    test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, get_val_transform(cfg), is_test=True)
    pin_memory = 'cuda' in cfg.DEVICE
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.EVAL_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    if not fold_info:
        raise ValueError('Fold 정보가 비어 있습니다. 학습을 먼저 진행하세요.')

    fold_f1s = [f1 for _, f1, _ in fold_info]
    weights = torch.tensor(fold_f1s, dtype=torch.float32)
    if weights.sum().item() == 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    avg_f1 = float(np.mean(fold_f1s))

    ensemble_logits = torch.zeros(len(test_dataset), cfg.NUM_CLASSES, dtype=torch.float32)
    use_amp = cfg.USE_AMP and 'cuda' in cfg.DEVICE

    for (fold_idx, (fold, f1, model_path)) in enumerate(fold_info):
        print(f'✅ Fold {fold} (F1: {f1:.4f}) 로드: {model_path}')
        model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=cfg.DEVICE)
        model.load_state_dict(state_dict)
        del state_dict
        model = model.to(cfg.DEVICE)
        model.eval()

        start_idx = 0
        with torch.no_grad():
            for images in tqdm(test_loader, desc=f'Inference | Fold {fold}'):
                images = images.to(cfg.DEVICE, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(images)
                    if cfg.TTA_HORIZONTAL_FLIP:
                        flipped = torch.flip(images, dims=[-1])
                        logits = (logits + model(flipped)) * 0.5

                batch_size = logits.size(0)
                ensemble_logits[start_idx:start_idx + batch_size] += logits.cpu() * weights[fold_idx].item()
                start_idx += batch_size

        del model
        if 'cuda' in cfg.DEVICE:
            torch.cuda.empty_cache()

    final_predictions = ensemble_logits.argmax(dim=1).numpy()
    return final_predictions.tolist(), avg_f1


# ==============================================================================
# 11. 제출 파일 생성
# ==============================================================================


def create_submission(test_df, predictions, avg_f1, exp_dir):
    filename = f'{exp_dir}/submission_{TIMESTAMP}_f1{avg_f1:.4f}.csv'

    submission = pd.DataFrame({'ID': test_df['ID'], 'target': predictions})
    submission.to_csv(filename, index=False)

    print(f"\n{'=' * 50}")
    print(f'제출 파일: {filename}')
    print(f"{'=' * 50}")
    print(submission.head(10))
    print('\n예측 분포:')
    print(submission['target'].value_counts().sort_index())

    return filename


# ==============================================================================
# 12. 메인 실행
# ==============================================================================


if __name__ == '__main__':
    download_and_extract_data()

    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    set_seed(config.SEED)

    EXP_DIR = f'experiments/exp_{TIMESTAMP}'
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(f'{EXP_DIR}/models', exist_ok=True)

    print('\n' + '=' * 70)
    print('🚀 문서 분류 학습 시작 (main1.py)')
    print('=' * 70)
    config.print_config()

    print('\n📂 데이터 로드 중...')
    train_df = pd.read_csv(f'{config.DATA_DIR}/train.csv')
    train_df['label'] = train_df['target']
    print(f'학습 데이터: {len(train_df)}장 | 클래스: {train_df.label.nunique()}개')

    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label'],
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), start=1):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        best_f1, model_path = train_fold(fold, train_fold_df, val_fold_df, EXP_DIR, class_weights, config)
        fold_results.append({'fold': fold, 'f1': best_f1, 'model_path': model_path})

    results_df = pd.DataFrame(fold_results)
    print(f"\n{'=' * 50}")
    print('📊 학습 결과')
    print(f"{'=' * 50}")
    print(results_df[['fold', 'f1']])
    print(f"\n평균 F1: {results_df['f1'].mean():.4f}")
    print(f"최고 F1: {results_df['f1'].max():.4f}")

    results_filename = f'{EXP_DIR}/fold_results_{TIMESTAMP}.csv'
    results_df.to_csv(results_filename, index=False)

    if os.path.exists(f'{config.DATA_DIR}/test.csv'):
        test_df = pd.read_csv(f'{config.DATA_DIR}/test.csv')
    else:
        test_df = pd.read_csv(f'{config.DATA_DIR}/sample_submission.csv').drop(columns=['target'])

    print(f'\n테스트 데이터: {len(test_df)}장')

    fold_info = [(row['fold'], row['f1'], row['model_path']) for _, row in results_df.iterrows()]
    predictions, avg_f1 = inference_ensemble(test_df, fold_info, config)
    submission_filename = create_submission(test_df, predictions, avg_f1, EXP_DIR)

    print('\n' + '=' * 70)
    print('✅ 모든 작업 완료!')
    print('=' * 70)
    print(f'📁 결과 파일: {results_filename}')
    print(f'📁 제출 파일: {submission_filename}')
    print('=' * 70 + '\n')

