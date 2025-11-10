"""학습된 ViT 모델로 제출 파일 생성"""
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

# 설정
class Config:
    DATA_DIR = 'data'
    TEST_DIR = 'data/test'
    MODEL_NAME = 'vit_base_patch16_384'
    IMG_SIZE = 384
    NUM_CLASSES = 17
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_TTA = False  # TTA 비활성화 (빠른 추론)

config = Config()

# Dataset
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ID']
        img_path = self.img_dir / img_id
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            img = self.transform(image=img)['image']

        return img

# Transform
def get_test_transform():
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# 추론
def inference(models, test_loader):
    all_preds = []

    for model in models:
        model.eval()
        preds_probs = []

        with torch.no_grad():
            for images in tqdm(test_loader, desc=f'Inference'):
                images = images.to(config.DEVICE)

                if config.USE_TTA:
                    # Original
                    outputs = model(images)
                    # Horizontal flip
                    outputs_flip = model(torch.flip(images, dims=[-1]))
                    outputs = (outputs + outputs_flip) / 2
                else:
                    outputs = model(images)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds_probs.append(probs)

        preds_probs = np.vstack(preds_probs)
        all_preds.append(preds_probs)

    # 앙상블 (평균)
    ensemble_probs = np.mean(all_preds, axis=0)
    final_preds = ensemble_probs.argmax(axis=1)

    return final_preds

# Main
if __name__ == '__main__':
    print("=" * 70)
    print("제출 파일 생성 시작")
    print("=" * 70)

    # 실험 디렉토리
    exp_dir = Path('experiments/exp_20251109_033413_vit90plus')

    # 테스트 데이터 로드 (test.csv가 없으면 디렉토리에서 직접 생성)
    test_csv_path = Path(f'{config.DATA_DIR}/test.csv')
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
    else:
        # test 폴더에서 이미지 목록 가져오기
        test_images = sorted([f.name for f in Path(config.TEST_DIR).glob('*.jpg')])
        test_df = pd.DataFrame({'ID': test_images})

    print(f"테스트 샘플 수: {len(test_df)}")

    # Dataset & DataLoader
    test_dataset = TestDataset(test_df, config.TEST_DIR, get_test_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 로드
    model_paths = sorted(exp_dir.glob('models/fold*.pth'))
    print(f"\n로드할 모델: {len(model_paths)}개")

    models = []
    fold_f1s = []

    for model_path in model_paths:
        # F1 스코어 추출
        f1_str = model_path.stem.split('_f1')[-1]
        f1 = float(f1_str)
        fold_f1s.append(f1)

        print(f"✅ {model_path.name} (F1: {f1:.4f})")

        # 모델 생성 및 로드
        model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=config.NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(config.DEVICE)
        models.append(model)

    avg_f1 = np.mean(fold_f1s)
    print(f"\n평균 F1: {avg_f1:.4f}")

    # 추론
    print("\n추론 시작...")
    predictions = inference(models, test_loader)

    # 제출 파일 생성
    timestamp = '20251109_033413'
    submission_path = exp_dir / f'submission_{timestamp}_f1{avg_f1:.4f}.csv'

    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })

    submission.to_csv(submission_path, index=False)

    print("\n" + "=" * 70)
    print(f"제출 파일 생성 완료!")
    print(f"위치: {submission_path}")
    print("=" * 70)
    print(submission.head(10))
    print(f"\n예측 분포:")
    print(submission['target'].value_counts().sort_index())
