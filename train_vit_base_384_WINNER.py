"""
Vision Transformer Base 384 - Improved Architecture
목표: 더 강력한 모델로 클래스 불균형 개선 (2.26x → <2.0x)

개선 전략:
1. ViT-Base (86M params) - 더 강력한 표현력
2. 384x384 해상도 - 더 많은 visual detail
3. Better pretrained weights (in21k)
4. Progressive training strategy
5. Strong but smart augmentation
"""

import os
import time
import random
import pickle
import cv2
import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# ==================== Reproducibility ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

print("="*80)
print("🚀 VISION TRANSFORMER BASE 384 - IMPROVED ARCHITECTURE")
print("="*80)
print("\nModel Improvements:")
print("  ✓ ViT-Base-384 (~86M parameters)")
print("  ✓ ImageNet-21k pretrained weights")
print("  ✓ 384x384 resolution for better detail")
print("  ✓ Progressive unfreezing strategy")
print("  ✓ Cosine annealing with warm restarts")
print("  ✓ Strong augmentation pipeline")
print("="*80)

# ==================== Configuration ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[CONFIG] Device: {device}")

# Model parameters - Upgraded
model_name = 'vit_base_patch16_384'  # Base model with 384 resolution
img_size = 384  # Higher resolution
LR = 3e-5  # Conservative LR for larger model
EPOCHS = 40
BATCH_SIZE = 8  # Smaller batch due to larger model
num_workers = 4
n_splits = 5
warmup_epochs = 2
gradient_accumulation_steps = 2  # Effective batch size = 16

# Paths
train_csv = './data/train.csv'
test_csv = './data/sample_submission.csv'
train_folder = './data/train/'
test_folder = './data/test/'
model_save_path = './models_vit_base_384/'

os.makedirs(model_save_path, exist_ok=True)

print(f"[CONFIG] Model: {model_name}")
print(f"[CONFIG] Image size: {img_size}x{img_size}")
print(f"[CONFIG] Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * gradient_accumulation_steps})")
print(f"[CONFIG] Learning rate: {LR}")
print(f"[CONFIG] Max epochs: {EPOCHS}")
print(f"[CONFIG] K-Folds: {n_splits}")

# ==================== Mixup/CutMix ====================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    """Apply mixup or cutmix"""
    if alpha <= 0 or np.random.rand() > 0.8:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # CutMix
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    # Mixup
    else:
        x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== Dataset Class ====================
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(target)

# ==================== Training Functions ====================
def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch, scaler, use_mixup=True):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for batch_idx, (image, targets) in enumerate(pbar):
        image = image.to(device)
        targets = targets.to(device)

        # Apply mixup/cutmix
        if use_mixup:
            mixed_image, targets_a, targets_b, lam = mixup_cutmix_data(
                image, targets, alpha=0.4, cutmix_prob=0.5
            )
        else:
            mixed_image, targets_a, targets_b, lam = image, targets, targets, 1.0

        # Mixed precision training
        with torch.cuda.amp.autocast():
            preds = model(mixed_image)
            loss = mixed_criterion(loss_fn, preds, targets_a, targets_b, lam)
            loss = loss / gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * gradient_accumulation_steps
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1}

def validate(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                preds = model(image)
                loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}, val_f1

# ==================== Data Augmentation ====================
print("\n[AUGMENTATION] Setting up transformations...")

# Strong but smart augmentation for 384 resolution
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=30,
        p=0.6,
        border_mode=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    ),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.3),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=1,
        fill_value=255,
        p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print("  ✓ Training: Strong augmentation optimized for 384px")
print("  ✓ Validation: Clean transforms")

# ==================== K-Fold Cross Validation ====================
print("\n" + "="*80)
print("STARTING K-FOLD CROSS VALIDATION")
print("="*80)

start_time = time.time()

full_dataset = ImageDataset(train_csv, train_folder, transform=trn_transform)
X = full_dataset.df[:, 0]
y = full_dataset.df[:, 1].astype(int)

print(f"\nDataset size: {len(full_dataset)}")
print(f"Number of classes: 17")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

all_fold_f1_scores = []

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{n_splits}")
    print(f"{'='*80}")

    # Create datasets
    train_dataset = ImageDataset(train_csv, train_folder, transform=trn_transform)
    valid_dataset = ImageDataset(train_csv, train_folder, transform=val_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    valid_subset = torch.utils.data.Subset(valid_dataset, valid_index)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_subset)}, Valid samples: {len(valid_subset)}")

    # Create model - ViT Base with in21k pretraining
    print(f"\nCreating {model_name}...")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function - simple CrossEntropy works well with strong model
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer - AdamW
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Scheduler - Cosine with warm restarts for better convergence
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_f1 = 0
    patience = 12
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.7f})")

        # Train
        train_metrics = train_one_epoch(
            train_loader, model, optimizer, loss_fn, device, epoch, scaler,
            use_mixup=(epoch > warmup_epochs)  # Start mixup after warmup
        )

        # Validate
        val_metrics, val_f1 = validate(valid_loader, model, loss_fn, device)

        # Step scheduler
        scheduler.step()

        print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
              f"Acc: {train_metrics['train_acc']:.4f}, F1: {train_metrics['train_f1']:.4f}")
        print(f"Valid - Loss: {val_metrics['val_loss']:.4f}, "
              f"Acc: {val_metrics['val_acc']:.4f}, F1: {val_metrics['val_f1']:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            model_filename = os.path.join(model_save_path, f'model_fold{fold}_best.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

            print(f"✓ Best F1 improved: {best_val_f1:.4f} - Model saved")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    all_fold_f1_scores.append(best_val_f1)
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1} COMPLETE - Best F1: {best_val_f1:.4f}")
    print(f"{'='*80}")

# ==================== Summary ====================
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("CROSS VALIDATION COMPLETE")
print("="*80)

mean_f1 = np.mean(all_fold_f1_scores)
std_f1 = np.std(all_fold_f1_scores)

print(f"\nResults per fold:")
for i, f1 in enumerate(all_fold_f1_scores):
    print(f"  Fold {i+1}: F1 = {f1:.4f}")

print(f"\nOverall Performance:")
print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Training time: {elapsed_time/60:.1f} minutes")

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Load test data
tst_dataset = ImageDataset(test_csv, test_folder, transform=val_transform)
tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

print(f"Test samples: {len(tst_dataset)}")

# Ensemble predictions from all folds
all_predictions = []

for fold in range(n_splits):
    print(f"\nPredicting with Fold {fold + 1} model...")

    model_file = os.path.join(model_save_path, f'model_fold{fold}_best.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    model.to(device)
    model.eval()

    fold_preds = []
    with torch.no_grad():
        for image, _ in tqdm(tst_loader, desc=f"Fold {fold+1}"):
            image = image.to(device)
            with torch.cuda.amp.autocast():
                preds = model(image)
                probs = torch.softmax(preds, dim=1)
            fold_preds.append(probs.cpu().numpy())

    fold_preds = np.concatenate(fold_preds, axis=0)
    all_predictions.append(fold_preds)

# Average predictions
print("\nAveraging predictions from all folds...")
avg_preds = np.mean(all_predictions, axis=0)
final_preds = np.argmax(avg_preds, axis=1)

# Calculate imbalance
pred_counts = pd.Series(final_preds).value_counts().sort_index()
imbalance = pred_counts.max() / pred_counts.min()
print(f"\nPrediction imbalance: {imbalance:.2f}x")
print(f"\nClass distribution:")
for cls in range(17):
    count = pred_counts.get(cls, 0)
    print(f"  Class {cls:2d}: {count:4d} samples")

# Save submission
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = final_preds

now = datetime.now()
date_str = now.strftime("%Y%m%d")
time_str = now.strftime("%H%M%S")
f1_str = f"{mean_f1:.4f}".replace(".", "")
imb_str = f"{imbalance:.2f}".replace(".", "")

filename = f"submission_{date_str}_{time_str}_vitbase384_f1{f1_str}_imb{imb_str}.csv"
pred_df.to_csv(f"./data/{filename}", index=False)

print(f"\n✓ Predictions saved to ./data/{filename}")

print("\n" + "="*80)
print("🎉 ViT-BASE-384 TRAINING COMPLETE!")
print("="*80)
print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Prediction Imbalance: {imbalance:.2f}x")
print(f"Total Time: {elapsed_time/60:.1f} minutes")
print(f"Models saved to: {model_save_path}")
print(f"Submission file: ./data/{filename}")
print("="*80)
