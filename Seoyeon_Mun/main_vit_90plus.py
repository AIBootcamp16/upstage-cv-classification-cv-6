"""Vision Transformer 기반 90점 돌파 (EfficientNet과 다른 접근)"""
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from datetime import datetime

# ========== 설정 (ViT 기반) ==========
class Config:
    def __init__(self):
        # 데이터
        self.DATA_DIR = 'data'
        self.TRAIN_DIR = 'data/train'
        self.TEST_DIR = 'data/test'

        # 모델 - Vision Transformer
        self.MODEL_NAME = 'vit_base_patch16_384'  # ViT-Base (384x384)
        self.IMG_SIZE = 384  # ViT는 고정 크기 (224 or 384)
        self.NUM_CLASSES = 17

        # 학습
        self.BATCH_SIZE = 8  # ViT는 메모리 많이 씀
        self.ACCUMULATION_STEPS = 2  # 효과적 배치 = 16
        self.EPOCHS = 15
        self.LR = 0.0001  # ViT는 낮은 lr
        self.N_FOLDS = 5

        # 정규화
        self.DROPOUT_RATE = 0.1  # ViT는 이미 dropout 내장
        self.WEIGHT_DECAY = 0.01
        self.LABEL_SMOOTHING = 0.1
        self.PATIENCE = 4

        # 증강
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.2

        # TTA
        self.USE_TTA = True
        self.TTA_FLIPS = [False, True]
        self.TTA_CROPS = ['center', 'topleft', 'topright', 'bottomleft', 'bottomright']

        # Stochastic Depth (DropPath) - ViT 전용
        self.DROP_PATH_RATE = 0.1

        self.SEED = 42
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

cfg = Config()
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ========== 데이터셋 ==========
class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['ID']}"
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        image = self.transform(image=image)['image']

        if self.is_test:
            return image
        else:
            label = row['target']
            return image, label

# ========== 증강 (ViT 최적화) ==========
train_transform = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),

    # ViT는 강한 증강에 robust
    A.Affine(
        translate_percent=0.1,
        scale=(0.85, 1.15),
        rotate=(-10, 10),
        p=0.6
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.CoarseDropout(
        max_holes=12,
        min_holes=4,
        max_height=24,
        max_width=24,
        min_height=8,
        min_width=8,
        p=0.4,
    ),

    # ViT에 효과적인 GridDropout
    A.GridDropout(ratio=0.3, p=0.3),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ========== MixUp ==========
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========== 학습 함수 ==========
def train_epoch(model, loader, criterion, optimizer, cfg):
    model.train()
    losses = []

    for images, labels in tqdm(loader, desc='Train'):
        images = images.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)

        optimizer.zero_grad()

        # MixUp
        if cfg.USE_MIXUP and np.random.random() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

def validate(model, loader, cfg):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Valid'):
            images = images.to(cfg.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(labels.numpy())

    f1 = f1_score(labels_list, preds_list, average='macro')
    return f1

# ========== 학습 실행 ==========
def train_fold(fold, train_loader, val_loader, exp_dir):
    print(f'\n{"="*70}')
    print(f'Fold {fold} 시작')
    print(f'{"="*70}')

    # 모델
    model = timm.create_model(
        cfg.MODEL_NAME,
        pretrained=True,
        num_classes=cfg.NUM_CLASSES,
        drop_rate=cfg.DROPOUT_RATE,
        drop_path_rate=cfg.DROP_PATH_RATE
    )
    model = model.to(cfg.DEVICE)
    print(f'✅ 모델: {cfg.MODEL_NAME}')

    # 옵티마이저 & 스케줄러 (ViT는 AdamW + warmup)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # Warmup + Cosine Annealing
    warmup_epochs = 2
    total_steps = cfg.EPOCHS * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LR,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # 학습
    best_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        # Train
        model.train()
        losses = []

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.EPOCHS}'):
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            optimizer.zero_grad()

            # MixUp
            if cfg.USE_MIXUP and np.random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        train_loss = np.mean(losses)
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

    # 저장
    model_filename = f'{exp_dir}/models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
    torch.save(best_model_state, model_filename)
    print(f'\nFold {fold} 완료 - F1: {best_f1:.4f}')

    return best_f1, model_filename

# ========== TTA 추론 (5-crop + flip) ==========
def inference_with_tta(test_df, fold_info, exp_dir):
    print(f'\n{"="*70}')
    print('ViT + TTA 앙상블 추론')
    print(f'{"="*70}')

    # 모델 로드
    models = []
    fold_f1s = []

    for fold, f1, model_path in fold_info:
        model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model = model.to(cfg.DEVICE)
        model.eval()
        models.append(model)
        fold_f1s.append(f1)
        print(f'✅ Fold {fold} (F1: {f1:.4f})')

    # 가중치
    weights = torch.tensor(fold_f1s, dtype=torch.float32)
    weights = weights / weights.sum()
    print(f'\n앙상블 가중치: {dict(zip([f[0] for f in fold_info], weights.tolist()))}')

    all_predictions = []

    if cfg.USE_TTA:
        print(f'\n🔄 TTA: {len(cfg.TTA_CROPS)} crops × {len(cfg.TTA_FLIPS)} flips = {len(cfg.TTA_CROPS) * len(cfg.TTA_FLIPS)} augmentations')

        # 5-crop transforms
        def get_crop_transform(crop_type, flip=False):
            if crop_type == 'center':
                transform = A.Compose([
                    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
                    A.HorizontalFlip(p=1.0 if flip else 0.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                # 약간 확대 후 crop
                crop_size = cfg.IMG_SIZE
                resize_size = int(cfg.IMG_SIZE * 1.1)

                if crop_type == 'topleft':
                    crop = A.Crop(x_min=0, y_min=0, x_max=crop_size, y_max=crop_size)
                elif crop_type == 'topright':
                    crop = A.Crop(x_min=resize_size-crop_size, y_min=0, x_max=resize_size, y_max=crop_size)
                elif crop_type == 'bottomleft':
                    crop = A.Crop(x_min=0, y_min=resize_size-crop_size, x_max=crop_size, y_max=resize_size)
                else:  # bottomright
                    crop = A.Crop(x_min=resize_size-crop_size, y_min=resize_size-crop_size, x_max=resize_size, y_max=resize_size)

                transform = A.Compose([
                    A.Resize(resize_size, resize_size),
                    crop,
                    A.HorizontalFlip(p=1.0 if flip else 0.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

            return transform

        tta_preds_all = []

        for crop_type in cfg.TTA_CROPS:
            for flip in cfg.TTA_FLIPS:
                print(f'  - Crop: {crop_type}, Flip: {flip}')

                tta_transform = get_crop_transform(crop_type, flip)
                test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, tta_transform, is_test=True)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

                tta_preds = []
                for images in tqdm(test_loader, desc=f'TTA {crop_type} flip={flip}'):
                    images = images.to(cfg.DEVICE)

                    fold_preds = []
                    for model in models:
                        with torch.no_grad():
                            pred = model(images)
                        fold_preds.append(pred.cpu())

                    # 앙상블
                    fold_preds_tensor = torch.stack(fold_preds)
                    weights_expanded = weights.unsqueeze(1).unsqueeze(2)
                    ensemble_pred = (fold_preds_tensor * weights_expanded).sum(dim=0)
                    tta_preds.append(ensemble_pred)

                tta_preds_all.append(torch.cat(tta_preds, dim=0))

        # TTA 평균
        final_preds = torch.stack(tta_preds_all).mean(dim=0)
        all_predictions = final_preds.argmax(dim=1).tolist()

    else:
        # TTA 없이
        test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, val_transform, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

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

            final_classes = ensemble_pred.argmax(dim=1).tolist()
            all_predictions.extend(final_classes)

    # 제출 파일
    avg_f1 = np.mean(fold_f1s)
    submission = test_df.copy()
    submission['target'] = all_predictions
    tta_suffix = '_tta' if cfg.USE_TTA else ''
    submission_filename = f'{exp_dir}/submission_{TIMESTAMP}_vit90plus{tta_suffix}_f1{avg_f1:.4f}.csv'
    submission.to_csv(submission_filename, index=False)

    print(f'\n✅ 제출 파일: {submission_filename}')

    # 분포
    pred_series = pd.Series(all_predictions)
    pred_dist = pred_series.value_counts().sort_index()
    pred_pct = (pred_dist / len(all_predictions) * 100).round(2)

    train_dist = {
        0: 6.37, 1: 2.93, 2: 6.37, 3: 6.37, 4: 6.37, 5: 6.37, 6: 6.37, 7: 6.37,
        8: 6.37, 9: 6.37, 10: 6.37, 11: 6.37, 12: 6.37, 13: 4.71, 14: 3.18, 15: 6.37, 16: 6.37
    }

    print(f'\n📊 예측 분포:')
    print(f'{"클래스":>6} | {"예측":>7} | {"예상":>7} | {"차이":>7}')
    print(f'{"-"*6}-+-{"-"*7}-+-{"-"*7}-+-{"-"*7}')
    total_mae = 0
    for cls in range(17):
        count = pred_dist.get(cls, 0)
        pct = pred_pct.get(cls, 0.0)
        expected = train_dist[cls]
        diff = pct - expected
        total_mae += abs(diff)
        sign = '+' if diff > 0 else ''
        print(f'  {cls:2d}   | {pct:5.2f}% | {expected:5.2f}% | {sign}{diff:5.2f}%')

    print(f'\n평균 절대 오차: {total_mae/17:.2f}%')

    return submission_filename

# ========== 메인 ==========
if __name__ == '__main__':
    # 실험 디렉토리
    EXP_DIR = f'experiments/exp_{TIMESTAMP}_vit90plus'
    os.makedirs(f'{EXP_DIR}/models', exist_ok=True)

    print('='*70)
    print('🚀 Vision Transformer 기반 90점 돌파')
    print('='*70)
    print(f'모델: {cfg.MODEL_NAME}')
    print(f'이미지 크기: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}')
    print(f'배치 크기: {cfg.BATCH_SIZE} × {cfg.ACCUMULATION_STEPS}')
    print(f'에폭: {cfg.EPOCHS}')
    print(f'학습률: {cfg.LR}')
    print(f'Drop Path Rate: {cfg.DROP_PATH_RATE}')
    print(f'MixUp: {cfg.USE_MIXUP}')
    print(f'TTA (5-crop + flip): {cfg.USE_TTA}')
    print(f'디바이스: {cfg.DEVICE}')
    print('='*70)

    # 데이터 로드
    train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')

    print(f'\n학습 데이터: {len(train_df)}개')
    print(f'테스트 데이터: {len(test_df)}개')

    # K-Fold 학습
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target']), 1):
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = DocumentDataset(train_fold_df, cfg.TRAIN_DIR, train_transform)
        val_dataset = DocumentDataset(val_fold_df, cfg.TRAIN_DIR, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

        f1, model_path = train_fold(fold, train_loader, val_loader, EXP_DIR)
        fold_results.append((fold, f1, model_path))

    # 결과 요약
    print(f'\n{"="*70}')
    print('학습 완료')
    print(f'{"="*70}')
    for fold, f1, path in fold_results:
        print(f'Fold {fold}: F1 {f1:.4f}')

    avg_f1 = np.mean([f1 for _, f1, _ in fold_results])
    print(f'\n평균 F1: {avg_f1:.4f}')

    # 추론
    submission_filename = inference_with_tta(test_df, fold_results, EXP_DIR)

    print(f'\n{"="*70}')
    print('✅ 모든 작업 완료!')
    print(f'{"="*70}')
