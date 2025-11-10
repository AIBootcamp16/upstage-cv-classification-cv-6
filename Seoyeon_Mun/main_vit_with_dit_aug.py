"""
ViT Base Model with DiT's Augmentation Strategy
For Document Classification Competition (17 classes)
With rotation/flip augmentation for upside-down documents
"""

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class CFG:
    # Paths
    train_dir = 'data/train'
    train_csv = 'data/train.csv'
    output_dir = 'experiments/vit_base_dit_aug_no_weights'

    # Model
    model_name = 'vit_base_patch16_384'
    num_classes = 17
    img_size = 384

    # Training
    n_folds = 5
    epochs = 30
    batch_size = 8
    num_workers = 4

    # Optimizer
    learning_rate = 5e-5
    weight_decay = 0.01

    # Scheduler
    warmup_epochs = 5
    min_lr = 1e-6

    # Class balancing
    use_class_weights = False

    # Other
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision = True

# ============================================================================
# Dataset
# ============================================================================
class DocumentDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(CFG.train_dir) / row['ID']

        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = row['target']
        return image, label

# ============================================================================
# Augmentation - DiT 전략 사용!
# ============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),

        # 문서 회전/뒤집기 (핵심!) - 뒤집힌 문서 대응
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),  # 90도 단위 회전 (0, 90, 180, 270)

        # 미세 회전 + 이동/스케일
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,  # ±15도 미세 회전
            border_mode=cv2.BORDER_CONSTANT,
            value=255,  # 흰색 패딩
            p=0.5
        ),

        # Perspective 왜곡 (스캔 문서 시뮬레이션)
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # Document-specific 노이즈
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),

        # Masking simulation (검은 박스)
        A.CoarseDropout(
            max_holes=8,
            max_height=80,
            max_width=80,
            min_holes=2,
            min_height=20,
            min_width=20,
            fill_value=0,
            p=0.5
        ),

        # 밝기/대비 (스캔 품질 차이)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),

        # 흑백 변환
        A.ToGray(p=0.1),

        # Grid distortion (문서 주름)
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{CFG.epochs} [Train]')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if CFG.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                         'lr': optimizer.param_groups[0]['lr']})

    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc='[Validation]')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    f1 = f1_score(all_labels, all_preds, average='macro')

    return running_loss / len(loader), f1

# ============================================================================
# Main Training Loop
# ============================================================================
def train_fold(fold, train_idx, valid_idx, df):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold+1}/{CFG.n_folds}")
    print(f"{'='*50}\n")

    # Data
    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]

    # Datasets
    train_dataset = DocumentDataset(train_df, get_train_transforms())
    valid_dataset = DocumentDataset(valid_df, get_valid_transforms())

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    # Model
    model = timm.create_model(
        CFG.model_name,
        pretrained=True,
        num_classes=CFG.num_classes
    )
    model.to(CFG.device)

    # Loss with class weights
    if CFG.use_class_weights:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(CFG.num_classes),
            y=train_df['target'].values
        )
        class_weights = torch.FloatTensor(class_weights).to(CFG.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    # Scheduler
    num_training_steps = len(train_loader) * CFG.epochs
    num_warmup_steps = len(train_loader) * CFG.warmup_epochs

    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return max(CFG.min_lr / CFG.learning_rate, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if CFG.mixed_precision else None

    # Training
    best_f1 = 0.0
    best_epoch = 0
    patience = 7
    patience_counter = 0
    best_model_path = None

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, CFG.device, epoch
        )
        valid_loss, valid_f1 = validate(model, valid_loader, criterion, CFG.device)

        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}")

        # Save best model (delete previous best)
        if valid_f1 > best_f1:
            # Delete previous best model
            if best_model_path is not None and Path(best_model_path).exists():
                Path(best_model_path).unlink()

            best_f1 = valid_f1
            best_epoch = epoch + 1
            patience_counter = 0

            Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
            best_model_path = f'{CFG.output_dir}/fold{fold+1}_f1{best_f1:.4f}.pth'
            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(f"✓ Best model saved! F1: {valid_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nFold {fold+1} Best F1: {best_f1:.4f} (Epoch {best_epoch})")

    return best_f1

# ============================================================================
# Main
# ============================================================================
def main():
    # Seed
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    # Load data
    df = pd.read_csv(CFG.train_csv)
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['target'].nunique()}")
    print(f"Class distribution:\n{df['target'].value_counts().sort_index()}")

    # K-Fold
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df['target'])):
        fold_f1 = train_fold(fold, train_idx, valid_idx, df)
        fold_scores.append(fold_f1)

    # Final Results
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold+1}: {score:.4f}")
    print(f"Average F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*50}\n")

    # Save results
    results_df = pd.DataFrame({
        'fold': range(1, CFG.n_folds + 1),
        'f1': fold_scores
    })
    results_df.to_csv(f'{CFG.output_dir}/fold_results.csv', index=False)
    print(f"Results saved to {CFG.output_dir}/fold_results.csv")

if __name__ == '__main__':
    main()
