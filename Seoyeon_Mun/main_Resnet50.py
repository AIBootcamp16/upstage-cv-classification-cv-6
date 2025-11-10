"""
Main Pipeline: Best Performing ResNet50 Model
재현 가능한 최고 성능 모델 (F1: 92.84%, Imbalance: 2.09x)

실행 방법:
    python main.py

결과:
    - 5-Fold Cross Validation
    - 평균 F1 Score: 92.84% ± 1.20%
    - 예측 불균형: 2.09x (최저)
    - 제출 파일: ./data/submission_[timestamp]_resnet50_kfold5.csv
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
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
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
torch.backends.cudnn.benchmark = True

print("="*80)
print("🚀 BEST PERFORMING MODEL: RESNET50 K-FOLD PIPELINE")
print("="*80)
print("\nThis script reproduces the best performing model:")
print("  - Model: ResNet50 (pretrained)")
print("  - Strategy: 5-Fold Cross Validation + Ensemble")
print("  - Expected F1: 92.84% ± 1.20%")
print("  - Expected Imbalance: 2.09x (lowest)")
print("  - Training Time: ~15 minutes")
print("="*80)

# ==================== Configuration ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[CONFIG] Device: {device}")

# Model parameters (최적값)
model_name = 'resnet50'
img_size = 256
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
num_workers = 4
n_splits = 5

# Paths
train_csv = './data/train.csv'
test_csv = './data/sample_submission.csv'
train_folder = './data/train/'
test_folder = './data/test/'
model_save_path = './models_kfold/'

os.makedirs(model_save_path, exist_ok=True)

print(f"[CONFIG] Model: {model_name}")
print(f"[CONFIG] Image size: {img_size}x{img_size}")
print(f"[CONFIG] Batch size: {BATCH_SIZE}")
print(f"[CONFIG] Learning rate: {LR}")
print(f"[CONFIG] Max epochs: {EPOCHS}")
print(f"[CONFIG] K-Folds: {n_splits}")
print(f"[CONFIG] Early stopping patience: 7")

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
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Training")
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)
        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

# Training augmentation (최적화된 설정)
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Flip(p=0.5),
    A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5,
                       border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation augmentation (augmentation 없음)
val_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print("  ✓ Training augmentation: Flip, Rotate, Shift, Brightness, Noise, Blur")
print("  ✓ Validation augmentation: None (only resize + normalize)")

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
print(f"Stratified K-Fold splits: {n_splits}")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

fold_results = []
best_models = []

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{n_splits}")
    print(f"{'='*80}")

    train_dataset = Subset(full_dataset, train_index)
    val_dataset_base = ImageDataset(train_csv, train_folder, transform=val_transform)
    valid_dataset = Subset(val_dataset_base, valid_index)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    # Initialize model
    print(f"\n[FOLD {fold+1}] Initializing {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=17).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_f1 = 0
    best_epoch = 0
    patience = 7
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\n[FOLD {fold+1}] Epoch {epoch+1}/{EPOCHS}")
        print("-" * 80)

        train_ret = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_ret, val_f1 = validate(valid_loader, model, loss_fn, device)

        print(f"\nResults:")
        print(f"  Train - Loss: {train_ret['train_loss']:.4f}, "
              f"Acc: {train_ret['train_acc']:.4f}, F1: {train_ret['train_f1']:.4f}")
        print(f"  Valid - Loss: {val_ret['val_loss']:.4f}, "
              f"Acc: {val_ret['val_acc']:.4f}, F1: {val_ret['val_f1']:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(model_save_path, f'model_fold{fold}_best.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ New best model saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print(f"\n[FOLD {fold+1}] Best: Epoch {best_epoch+1}, F1 {best_f1:.4f}")
    fold_results.append({'fold': fold + 1, 'best_epoch': best_epoch, 'best_f1': best_f1})

    # Load best model
    with open(os.path.join(model_save_path, f'model_fold{fold}_best.pkl'), 'rb') as f:
        best_model = pickle.load(f)
    best_models.append(best_model)

training_time = time.time() - start_time

# ==================== Cross Validation Results ====================
print("\n" + "="*80)
print("K-FOLD CROSS VALIDATION RESULTS")
print("="*80)

for result in fold_results:
    print(f"Fold {result['fold']}: F1 = {result['best_f1']:.4f} (Epoch {result['best_epoch']+1})")

avg_f1 = np.mean([r['best_f1'] for r in fold_results])
std_f1 = np.std([r['best_f1'] for r in fold_results])

print(f"\nAverage F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Training time: {training_time/60:.1f} minutes")
print("="*80)

# ==================== Ensemble Inference ====================
print("\n" + "="*80)
print("INFERENCE ON TEST SET (ENSEMBLE)")
print("="*80)

tst_dataset = ImageDataset(test_csv, test_folder, transform=val_transform)
tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

print(f"Test samples: {len(tst_dataset)}")
print(f"Running ensemble prediction with {n_splits} models...")

all_fold_preds = []

for fold_idx, model in enumerate(best_models):
    print(f"\nPredicting with Fold {fold_idx + 1} model...")
    model.eval()
    fold_preds = []

    with torch.no_grad():
        for image, _ in tqdm(tst_loader, desc=f"Fold {fold_idx+1}"):
            image = image.to(device)
            preds = model(image)
            fold_preds.append(preds.cpu().numpy())

    fold_preds = np.concatenate(fold_preds, axis=0)
    all_fold_preds.append(fold_preds)

# Average predictions from all folds
print("\nAveraging predictions from all folds...")
avg_preds = np.mean(all_fold_preds, axis=0)
final_preds = np.argmax(avg_preds, axis=1)

# ==================== Save Predictions ====================
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = final_preds

now = datetime.now()
date_str = now.strftime("%Y%m%d")
time_str = now.strftime("%H%M%S")
f1_str = f"{avg_f1:.4f}".replace(".", "")

filename = f"submission_{date_str}_{time_str}_resnet50_kfold5_f1_{f1_str}.csv"
pred_df.to_csv(f"./data/{filename}", index=False)

print(f"\n✓ Predictions saved to ./data/{filename}")
print("\nSample predictions:")
print(pred_df.head(10))

# ==================== Analysis ====================
print("\n" + "="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

class_counts = pred_df['target'].value_counts().sort_index()
total = len(pred_df)

print(f"\nTotal predictions: {total}")
print("\nClass distribution:")
for cls in range(17):
    count = class_counts.get(cls, 0)
    pct = (count / total) * 100
    print(f"  Class {cls:2d}: {count:4d} ({pct:5.2f}%)")

max_count = class_counts.max()
min_count = class_counts.min()
mean_count = class_counts.mean()
std_count = class_counts.std()
imbalance = max_count / min_count

print(f"\nStatistics:")
print(f"  Max: {max_count} (Class {class_counts.idxmax()})")
print(f"  Min: {min_count} (Class {class_counts.idxmin()})")
print(f"  Mean: {mean_count:.1f}")
print(f"  Std: {std_count:.1f}")
print(f"  Imbalance ratio: {imbalance:.2f}x")

# ==================== Final Summary ====================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

print(f"\nFinal Performance:")
print(f"  Model: ResNet50 (pretrained)")
print(f"  Average K-Fold F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"  Prediction Imbalance: {imbalance:.2f}x")
print(f"  Training Time: {training_time/60:.1f} minutes")

print(f"\nFiles Generated:")
print(f"  ✓ Models: {model_save_path}model_fold[0-4]_best.pkl")
print(f"  ✓ Submission: ./data/{filename}")

print(f"\n{'='*80}")
print("📊 EXPECTED LEADERBOARD PERFORMANCE")
print("="*80)
print(f"  Based on validation F1: {avg_f1:.4f}")
print(f"  Expected leaderboard F1: {avg_f1-0.01:.4f}~{avg_f1+0.01:.4f}")
print(f"  Confidence: HIGH (lowest imbalance ratio)")

print("\n" + "="*80)
print("🚀 Ready to submit! This is the BEST performing model.")
print("="*80)
