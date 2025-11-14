"""
Mega Pipeline: Single-Run High-Performance Submission Generator
===============================================================
목표:
 - 제출 1회에: (선택된 여러 모델 학습 or 기존 체크포인트 로드) → OOF 생성 → TTA(Test) → 가중치 최적화 → 앙상블 → 단일 제출 CSV 생성
 - 10회 제출 안에 0.95 이상, 최종 0.97 접근 전략 기반 코어 인프라 제공

핵심 기능:
 1. 모델 팩토리: ViT-Base, ViT-Large, ConvNeXt-Large, Swin-Large(추가 가능) 등
 2. 유연한 설정: 신규 학습 / 기존 체크포인트 사용 선택 (존재 시 자동 로드)
 3. 고급 학습 기법: Mixup/CutMix, Gradient Accumulation, EMA(Optional), SWA(Optional)
 4. 강력한 증강: Base / Strong / Resolution Multi-Scale 조합 선택
 5. TTA: 다양한 transform + multi-scale (예: 352, 384, 416) + flips
 6. 앙상블 가중치 최적화: Dirichlet 샘플링 + 국소 탐색 (OOF macro F1 기준)
 7. Calibration (Temperature Scaling 후) 옵션
 8. 로그 및 중간 산출물 저장 (./pipeline_artifacts)

사용 방법 (예시):
  python mega_pipeline.py \
    --models vit_large,convnext_large,vit_base_strong \
    --train True \
    --epochs 30 \
    --tta original,hflip,vflip,rotate90 \
    --tta-multiscales 352,384,416 \
    --opt-iters 200 \
    --swa True --ema True

실행 결과:
  ./data/submission_YYYYMMDD_HHMMSS_megapipeline_f1XXXX.csv 생성

주의:
 - 실제 0.97 달성 위해서는 추가적인 아키텍처(예: EVA, BEiT, DeiT-III), 더 긴 학습, 외부 pretraining, 데이터 클리닝 필요.
 - 본 스크립트는 확장성과 재사용성 우선.
"""

import os
import sys
import math
import time
import json
import argparse
import random
import pickle
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# =============================================================
# Reproducibility
# =============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================
# Dataset
# =============================================================
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, folder: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.folder = folder
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.folder, row['ID'])))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(row['target']) if 'target' in row else -1

# =============================================================
# Augmentations
# =============================================================
def get_transforms(style: str, img_size: int):
    if style == 'base':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6, border_mode=0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
            ], p=0.5),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    elif style == 'strong':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.9, border_mode=0),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=50, p=0.7, border_mode=0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10,50), p=1.0),
                A.GaussianBlur(blur_limit=(3,7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0)
            ], p=0.5),
            A.CoarseDropout(max_holes=12, max_height=40, max_width=40, p=0.5),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f'Unknown transform style: {style}')

def get_val_transform(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

# =============================================================
# Mixup / CutMix
# =============================================================
def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_cutmix_data(x, y, alpha=0.4, cutmix_prob=0.5):
    if alpha <= 0 or np.random.rand() > 0.9:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    else:
        x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================
# Model Factory
# =============================================================
def create_model(model_tag: str, num_classes: int):
    if model_tag == 'vit_base':
        return timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
    if model_tag == 'vit_base_strong':
        return timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.3)
    if model_tag == 'vit_large':
        return timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
    if model_tag == 'convnext_large':
        return timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True, num_classes=num_classes, drop_rate=0.2)
    if model_tag == 'swin_large':
        return timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes, drop_rate=0.2)
    raise ValueError(f'Unknown model tag: {model_tag}')

# =============================================================
# Training & Validation
# =============================================================
def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch, accumulation_steps, use_mixup, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Train E{epoch}')
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        if use_mixup:
            images, ta, tb, lam = mixup_cutmix_data(images, targets, alpha=0.4, cutmix_prob=0.5)
        else:
            ta, tb, lam = targets, targets, 1.0
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = mixed_criterion(loss_fn, out, ta, tb, lam) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        preds_all.append(out.argmax(1).detach().cpu().numpy())
        targets_all.append(targets.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item()*accumulation_steps:.4f}'})
    preds_cat = np.concatenate(preds_all)
    targets_cat = np.concatenate(targets_all)
    acc = accuracy_score(targets_cat, preds_cat)
    f1 = f1_score(targets_cat, preds_cat, average='macro')
    return total_loss / len(loader), acc, f1

@torch.no_grad()
def validate(loader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out, targets)
        total_loss += loss.item()
        preds_all.append(out.argmax(1).cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    preds_cat = np.concatenate(preds_all)
    targets_cat = np.concatenate(targets_all)
    acc = accuracy_score(targets_cat, preds_cat)
    f1 = f1_score(targets_cat, preds_cat, average='macro')
    return total_loss / len(loader), acc, f1

# =============================================================
# OOF & Test Prediction (with TTA)
# =============================================================
@torch.no_grad()
def predict_loader(model, loader, return_logits: bool = False):
    model.eval()
    probs_all = []
    logits_all = []
    for images, _ in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        if return_logits:
            logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
    probs_all = np.concatenate(probs_all, axis=0)
    if return_logits:
        logits_all = np.concatenate(logits_all, axis=0)
        return probs_all, logits_all
    return probs_all

# Temperature Scaling (Calibration)
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(logits, targets, max_iter=50, lr=0.01):
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=100)
    nll = nn.CrossEntropyLoss()
    logits_t = torch.from_numpy(logits).float().to(device)
    targets_t = torch.from_numpy(targets).long().to(device)
    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits_t), targets_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        calibrated = torch.softmax(scaler(logits_t), dim=1).cpu().numpy()
    return calibrated, scaler.temperature.item()

# =============================================================
# Ensemble Weight Optimization
# =============================================================
def _softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def _project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Project v onto the capped simplex {w >= 0, sum w = 1, w_i <= cap}."""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    # Handle trivial cases
    if cap >= 1.0:
        v = np.maximum(v, 0)
        s = v.sum()
        return v / s if s > 0 else np.ones_like(v) / n
    # Bisection on tau for y_i = clip(v_i - tau, 0, cap), sum y_i = 1
    # Set bounds for tau
    lo = -1e3
    hi = 1e3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        y = v - mid
        y = np.clip(y, 0.0, cap)
        s = y.sum()
        if s > 1.0:
            # Need larger tau to shrink y
            lo = mid
        else:
            hi = mid
    y = v - hi
    y = np.clip(y, 0.0, cap)
    # Normalize tiny numerical drift
    s = y.sum()
    if s <= 0:
        y = np.ones_like(v) / n
    else:
        y = y / s
    return y.astype(np.float32)

def optimize_weights(oof_list: List[np.ndarray], y_true: np.ndarray, n_iter=200, seed=42, space: str = 'prob', cap: float | None = None):
    rng = np.random.default_rng(seed)
    best = {'w': np.array([1/len(oof_list)]*len(oof_list)), 'f1': -1}
    def eval_w(w):
        if space == 'prob':
            blend = sum(w[i]*oof_list[i] for i in range(len(oof_list)))
            preds = blend.argmax(1)
        elif space == 'logit':
            blend_logits = sum(w[i]*oof_list[i] for i in range(len(oof_list)))
            preds = _softmax_np(blend_logits, axis=1).argmax(1)
        else:
            raise ValueError(f'Unknown ensemble space: {space}')
        return f1_score(y_true, preds, average='macro')
    # Dirichlet sampling
    for _ in range(n_iter):
        w = rng.dirichlet([1.0]*len(oof_list))
        if cap is not None:
            w = _project_capped_simplex(w, cap)
        f1 = eval_w(w)
        if f1 > best['f1']:
            best = {'w': w, 'f1': f1}
    # Local refinement (coordinate ascent)
    w = best['w'].copy()
    for _ in range(50):
        improved = False
        for i in range(len(w)):
            for delta in [-0.05, 0.05]:
                w2 = w.copy()
                w2[i] = np.clip(w2[i] + delta, 0, 1)
                if w2.sum() == 0: 
                    continue
                w2 /= w2.sum()
                if cap is not None:
                    w2 = _project_capped_simplex(w2, cap)
                f1 = eval_w(w2)
                if f1 > best['f1']:
                    best = {'w': w2, 'f1': f1}
                    w = w2
                    improved = True
        if not improved:
            break
    return best['w'], best['f1']

# =============================================================
# Pipeline Execution
# =============================================================
def run_pipeline(args):
    os.makedirs('pipeline_artifacts', exist_ok=True)
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/sample_submission.csv')
    num_classes = train_df['target'].nunique()

    # Class Weights (balanced inverse freq)
    counts = train_df['target'].value_counts().sort_index().values
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # CV Split
    X = train_df['ID'].values
    y = train_df['target'].values.astype(int)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    # Storage
    oof_preds_models = {m: np.zeros((len(train_df), num_classes), dtype=np.float32) for m in args.models}
    test_preds_models = {m: [] for m in args.models}
    # For logit-space ensembling
    oof_logits_for_ens = {m: np.zeros((len(train_df), num_classes), dtype=np.float32) for m in args.models}
    test_logits_models = {m: [] for m in args.models}

    # For calibration storage
    logits_oof_models = {m: np.zeros((len(train_df), num_classes), dtype=np.float32) for m in args.models}

    for model_tag in args.models:
        print('\n' + '='*90)
        print(f'Model: {model_tag}')
        print('='*90)
        transform_style = 'strong' if 'strong' in model_tag else 'base'
        trn_transform = get_transforms(transform_style, args.img_size)
        val_transform = get_val_transform(args.img_size)

        # Per-model TTA scales (ViT models typically require fixed 384)
        if 'vit' in model_tag:
            model_scales = [args.img_size]
        else:
            model_scales = args.tta_multiscales

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            print(f'\nFOLD {fold+1}/{args.folds}')
            fold_train_df = train_df.iloc[tr_idx]
            fold_valid_df = train_df.iloc[va_idx]

            train_dataset = ImageDataset(fold_train_df, './data/train', trn_transform)
            valid_dataset = ImageDataset(fold_valid_df, './data/train', val_transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            # Resolve checkpoint path (priority: user supplied --ckpt-map)
            if model_tag in getattr(args, 'ckpt_map', {}):
                ckpt_path = args.ckpt_map[model_tag].format(fold=fold)
                ckpt_dir = os.path.dirname(ckpt_path)
                if not os.path.isfile(ckpt_path) and not args.train:
                    raise FileNotFoundError(f'Checkpoint not found for model {model_tag} fold {fold}: {ckpt_path}. Enable --train True or supply correct path in --ckpt-map.')
            else:
                ckpt_dir = f'./pipeline_artifacts/{model_tag}'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'{model_tag}_fold{fold}.pkl')

            if args.train and (not os.path.isfile(ckpt_path)):
                model = create_model(model_tag, num_classes).to(device)
                loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=torch.FloatTensor(class_weights).to(device))
                optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)
                scaler = torch.cuda.amp.GradScaler()
                best_f1 = -1
                patience_cnt = 0
                for epoch in range(1, args.epochs+1):
                    use_mix = epoch > args.mixup_warmup
                    train_loss, train_acc, train_f1 = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch, args.accumulation, use_mix)
                    val_loss, val_acc, val_f1 = validate(valid_loader, model, loss_fn)
                    scheduler.step()
                    print(f'Epoch {epoch}: train_f1={train_f1:.4f} val_f1={val_f1:.4f}')
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        patience_cnt = 0
                        with open(ckpt_path, 'wb') as f:
                            pickle.dump(model, f)
                        print(f'  ✓ New best F1 {best_f1:.4f} saved.')
                    else:
                        patience_cnt += 1
                        if patience_cnt >= args.patience:
                            print('  Early stopping.')
                            break
                del model
                torch.cuda.empty_cache()
            else:
                print(f'  Loading existing checkpoint: {ckpt_path}')

            # Load best model
            with open(ckpt_path, 'rb') as f:
                model = pickle.load(f).to(device)

            # OOF predictions (logits & probs)
            model.eval()
            oof_logits_list = []
            oof_probs_list = []
            for images, targets in DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers):
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                oof_logits_list.append(logits.detach().cpu().numpy())
                oof_probs_list.append(probs.detach().cpu().numpy())
            oof_logits = np.concatenate(oof_logits_list, axis=0)
            oof_probs = np.concatenate(oof_probs_list, axis=0)

            oof_preds_models[model_tag][va_idx] = oof_probs
            logits_oof_models[model_tag][va_idx] = oof_logits
            oof_logits_for_ens[model_tag][va_idx] = oof_logits

            # TTA Test predictions (fold)
            fold_test_preds_accum = []
            fold_test_logits_accum = []
            for scale in model_scales:
                for tname in args.tta:
                    t = build_tta_transform(tname, scale)
                    test_ds = ImageDataset(test_df[['ID']].assign(target=-1), './data/test', t)
                    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                    if args.ensemble_space == 'logit':
                        probs, logits_batch = predict_loader(model, test_loader, return_logits=True)
                        fold_test_logits_accum.append(logits_batch)
                    else:
                        probs = predict_loader(model, test_loader, return_logits=False)
                    fold_test_preds_accum.append(probs)
            # Average across TTA variants for this fold
            fold_test_mean = np.mean(fold_test_preds_accum, axis=0)
            test_preds_models[model_tag].append(fold_test_mean)
            if args.ensemble_space == 'logit' and fold_test_logits_accum:
                fold_test_logits_mean = np.mean(fold_test_logits_accum, axis=0)
                test_logits_models[model_tag].append(fold_test_logits_mean)

            del model
            torch.cuda.empty_cache()

        # After folds: average test predictions across folds
        test_preds_models[model_tag] = np.mean(test_preds_models[model_tag], axis=0)
        if args.ensemble_space == 'logit' and test_logits_models[model_tag]:
            test_logits_models[model_tag] = np.mean(test_logits_models[model_tag], axis=0)

    # Calibration (optional)
    if args.calibrate:
        print('\nCalibration (Temperature Scaling)...')
        for m in args.models:
            logits_all = logits_oof_models[m]
            targets_all = y
            calibrated_probs, temp = calibrate_temperature(logits_all, targets_all)
            oof_preds_models[m] = calibrated_probs
            print(f'  Model {m}: temperature={temp:.3f}')

    # Optimize Ensemble Weights
    print('\nOptimizing ensemble weights...')
    if args.ensemble_space == 'logit':
        oof_list = [oof_logits_for_ens[m] for m in args.models]
        weights, best_oof_f1 = optimize_weights(oof_list, y, n_iter=args.opt_iters, seed=SEED, space='logit', cap=(args.weight_cap if args.weight_cap < 1.0 else None))
    else:
        oof_list = [oof_preds_models[m] for m in args.models]
        weights, best_oof_f1 = optimize_weights(oof_list, y, n_iter=args.opt_iters, seed=SEED, space='prob', cap=(args.weight_cap if args.weight_cap < 1.0 else None))
    print(f'Best OOF F1={best_oof_f1:.5f} | weights=' + ','.join(f'{w:.3f}' for w in weights))

    # Blend test in selected space
    if args.ensemble_space == 'logit' and all(isinstance(test_logits_models[m], np.ndarray) for m in args.models):
        blended_test_logits = sum(weights[i]*test_logits_models[args.models[i]] for i in range(len(args.models)))
        blended_test = _softmax_np(blended_test_logits, axis=1)
    else:
        blended_test = sum(weights[i]*test_preds_models[args.models[i]] for i in range(len(args.models)))

    # Optional: class prior alignment to training distribution
    prior_info = None
    if args.prior_align != 'off':
        # target prior
        train_counts = train_df['target'].value_counts().sort_index().values.astype(np.float64)
        pi_target = train_counts / train_counts.sum()
        # current predicted prior
        pi_pred = blended_test.mean(axis=0)
        # avoid division by zero
        eps = 1e-8
        ratio = (pi_target + eps) / (pi_pred + eps)
        ratio = ratio ** args.prior_alpha
        adjusted = blended_test * ratio[None, :]
        adjusted = adjusted / (adjusted.sum(axis=1, keepdims=True) + eps)
        blended_test = adjusted
        prior_info = {
            'pi_target': pi_target.tolist(),
            'pi_pred_before': pi_pred.tolist(),
            'alpha': args.prior_alpha
        }

    final_preds = blended_test.argmax(1)

    # Submission
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    f1_str = f'{best_oof_f1:.4f}'.replace('.', '')
    submission_name = f'submission_{date_str}_{time_str}_megapipeline_f1{f1_str}.csv'
    out_path = os.path.join('./data', submission_name)
    pd.DataFrame({'ID': test_df['ID'], 'target': final_preds}).to_csv(out_path, index=False)

    # Class distribution
    counts = pd.Series(final_preds).value_counts().sort_index()
    imbalance = counts.max() / counts.min()

    report = {
        'models': args.models,
        'weights': [float(w) for w in weights],
        'oof_f1': float(best_oof_f1),
        'submission': submission_name,
        'class_distribution': counts.to_dict(),
        'imbalance': float(imbalance),
        'tta': args.tta,
        'tta_multiscales': args.tta_multiscales,
        'calibrated': args.calibrate,
        'ensemble_space': args.ensemble_space,
        'prior_align': args.prior_align,
        'prior_info': prior_info,
    }
    with open('./pipeline_artifacts/megapipeline_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('\n' + '='*90)
    print('FINAL SUBMISSION GENERATED')
    print('='*90)
    print(f'Submission file: {out_path}')
    print(f'OOF Macro F1: {best_oof_f1:.5f}')
    print(f'Weights: ' + ', '.join(f'{m}:{w:.3f}' for m, w in zip(args.models, weights)))
    print(f'Class imbalance: {imbalance:.2f}x')
    print('Class distribution:')
    for cls in range(num_classes):
        print(f'  Class {cls:2d}: {counts.get(cls,0):4d}')
    print('='*90)

# =============================================================
# TTA Transform Builder
# =============================================================
def build_tta_transform(name: str, img_size: int):
    base = [A.Resize(img_size, img_size)]
    if name == 'original':
        pass
    elif name == 'hflip':
        base.append(A.HorizontalFlip(p=1.0))
    elif name == 'vflip':
        base.append(A.VerticalFlip(p=1.0))
    elif name == 'rotate90':
        base.append(A.Rotate(limit=(90,90), p=1.0))
    else:
        raise ValueError(f'Unknown TTA transform {name}')
    base.append(A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)))
    base.append(ToTensorV2())
    return A.Compose(base)

# =============================================================
# Args
# =============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=str, default='vit_large,convnext_large,vit_base_strong', help='Comma separated model tags')
    p.add_argument('--train', type=str, default='False', help='Train models (True/False) or just load existing checkpoints')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--mixup-warmup', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--tta', type=str, default='original,hflip,vflip')
    p.add_argument('--tta-multiscales', type=str, default='384')
    p.add_argument('--opt-iters', type=int, default=200)
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--ckpt-map', type=str, default='', help='Semicolon separated model_tag=path_pattern entries using {fold}. Example: "vit_large=/root/git/upstage-cv-classification-cv-6/Hyun_Choi/models_vit_large_384_weighted/model_fold{fold}_best.pkl"')
    p.add_argument('--ensemble-space', dest='ensemble_space', type=str, default='prob', choices=['prob','logit'], help='Blend probabilities or logits')
    p.add_argument('--prior-align', dest='prior_align', type=str, default='off', choices=['off','train'], help='Apply class prior alignment on test predictions')
    p.add_argument('--prior-alpha', dest='prior_alpha', type=float, default=0.5, help='Strength of prior alignment (0=no-op, 1=full)')
    p.add_argument('--weight-cap', dest='weight_cap', type=float, default=1.0, help='Upper cap for each ensemble weight (<=1.0). 1.0 means no cap')
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.tta = [t.strip() for t in args.tta.split(',') if t.strip()]
    args.tta_multiscales = [int(s) for s in args.tta_multiscales.split(',') if s.strip()]
    args.train = args.train.lower() == 'true'
    # parse ckpt-map string into dict
    ckpt_map = {}
    if hasattr(args, 'ckpt_map') and args.ckpt_map.strip():
        # support both ';' and ',' as separators to avoid shell-semicolon issues
        raw = args.ckpt_map.replace(',', ';')
        entries = [e for e in raw.split(';') if e.strip()]
        for entry in entries:
            if '=' not in entry:
                raise ValueError(f'Invalid ckpt-map entry (missing =): {entry}')
            k, v = entry.split('=', 1)
            ckpt_map[k.strip()] = v.strip()
    args.ckpt_map = ckpt_map
    return args

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
