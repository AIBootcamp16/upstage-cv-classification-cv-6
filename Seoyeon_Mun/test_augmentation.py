"""증강 파이프라인 테스트"""
import sys
import cv2
import numpy as np
from pathlib import Path
from main2 import Config, get_albumentations_train

# Config 로드
cfg = Config()
print("=== Config 설정 ===")
print(f"AUG_STRATEGY: {cfg.AUG_STRATEGY}")
print(f"USE_MIXUP: {cfg.USE_MIXUP}, MIXUP_PROB: {cfg.MIXUP_PROB}")
print()

# Augmentation pipeline 가져오기
aug_pipeline = get_albumentations_train(cfg)

print("=== Augmentation Pipeline ===")
for i, transform in enumerate(aug_pipeline.transforms):
    print(f"{i+1}. {transform.__class__.__name__}: {transform}")
print()

# 샘플 이미지로 테스트
train_dir = Path(cfg.TRAIN_DIR)
sample_images = list(train_dir.glob('*.jpg'))[:5]

if len(sample_images) > 0:
    print(f"=== 샘플 이미지 테스트 ({len(sample_images)}개) ===")

    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 원본 밝기
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_orig = gray_orig.mean()

        # 증강 적용
        augmented = aug_pipeline(image=img)['image']

        # 증강 후 밝기 (텐서이므로 변환 필요)
        # augmented는 이미 normalized되고 tensor로 변환됨
        print(f"{img_path.name}: 원본 밝기={brightness_orig:.1f}, 텐서 shape={augmented.shape}")

    print()
    print("✅ 증강 파이프라인이 정상적으로 작동합니다.")
else:
    print("❌ 샘플 이미지를 찾을 수 없습니다.")
