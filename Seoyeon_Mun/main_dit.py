"""
Document Image Transformer (DiT) Large - Training Script
For Document Classification Competition (17 classes)
With Rotation/Flip Augmentation for upside-down documents
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class CFG:
    # Paths
    train_dir = 'data/train'  # 학습 이미지 폴더
    train_csv = 'data/train.csv'  # 'ID', 'target' 컬럼 필요
    output_dir = 'experiments/dit_large_384_models'
    
    # Model
    model_name = 'microsoft/dit-large'
    num_classes = 17
    img_size = 384  # 224 → 384 (성능 향상)

    # Training
    n_folds = 5
    epochs = 30
    batch_size = 8  # 16 → 8 (img_size 증가로 메모리 고려)
    num_workers = 4

    # Optimizer
    learning_rate = 5e-5  # 3e-5 → 5e-5 (약간 높여서 더 빠른 학습)
    weight_decay = 0.01

    # Scheduler
    warmup_epochs = 5  # 3 → 5 (더 안정적인 시작)
    
    # Other
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision = True  # FP16 학습 (속도 향상)

# ============================================================================
# Dataset
# ============================================================================
class DocumentDataset(Dataset):
    def __init__(self, df, transform=None, processor=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.processor = processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(CFG.train_dir) / row['ID']  # ID에 이미 확장자 포함
        
        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Processor (DiT용)
        if self.processor:
            # Albumentations 후 numpy array를 PIL로 변환
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        else:
            pixel_values = image
        
        label = row['target']

        return pixel_values, label

# ============================================================================
# Augmentation - 회전/뒤집기 강화!
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
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
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
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).logits
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
        
        outputs = model(images).logits
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
    
    # Processor
    processor = AutoImageProcessor.from_pretrained(CFG.model_name)
    
    # Datasets
    train_dataset = DocumentDataset(train_df, get_train_transforms(), processor)
    valid_dataset = DocumentDataset(valid_df, get_valid_transforms(), processor)
    
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
    model = AutoModelForImageClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(CFG.device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * CFG.epochs
    num_warmup_steps = len(train_loader) * CFG.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if CFG.mixed_precision else None
    
    # Training
    best_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, CFG.device, epoch
        )
        valid_loss, valid_f1 = validate(model, valid_loader, criterion, CFG.device)
        
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}")
        
        # Save best model
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': valid_f1,
            }, f'{CFG.output_dir}/dit_large_fold{fold}_best.pth')
            
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

if __name__ == '__main__':
    main()