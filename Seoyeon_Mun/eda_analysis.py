import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from tqdm import tqdm

class DocumentEDA:
    def __init__(self, data_dir='/root/upstage-cv-classification-cv-6/data'):
        self.data_dir = Path(data_dir)
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.meta_df = pd.read_csv(self.data_dir / 'meta.csv')
        self.test_dir = self.data_dir / 'test'
        self.train_dir = self.data_dir / 'train'

        self.results = {
            'train': {},
            'test': {},
            'recommendations': []
        }

    def analyze_class_distribution(self):
        """클래스 분포 분석"""
        print("=" * 60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)

        class_counts = self.train_df['target'].value_counts().sort_index()

        for idx, count in class_counts.items():
            class_name = self.meta_df[self.meta_df['target'] == idx]['class_name'].values[0]
            print(f"Class {idx:2d} ({class_name:50s}): {count:4d} images")

        print(f"\nTotal training images: {len(self.train_df)}")
        print(f"Min samples per class: {class_counts.min()}")
        print(f"Max samples per class: {class_counts.max()}")
        print(f"Mean samples per class: {class_counts.mean():.1f}")
        print(f"Std samples per class: {class_counts.std():.1f}")

        self.results['train']['class_distribution'] = class_counts.to_dict()

        # 불균형 체크
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 2:
            self.results['recommendations'].append(
                f"HIGH CLASS IMBALANCE (ratio {imbalance_ratio:.2f}:1) - Consider class weights or balanced sampling"
            )

    def analyze_image_properties(self, split='train', sample_size=None):
        """이미지 속성 분석 (크기, 종횡비, 색상 등)"""
        print("\n" + "=" * 60)
        print(f"{split.upper()} IMAGE PROPERTIES ANALYSIS")
        print("=" * 60)

        if split == 'train':
            image_ids = self.train_df['ID'].values
            img_dir = self.train_dir
            ext = ''  # ID already includes .jpg
        else:
            image_ids = [f.name for f in self.test_dir.glob('*.jpg')]  # Use .name to include extension
            img_dir = self.test_dir
            ext = ''  # filename already includes .jpg

        if sample_size and sample_size < len(image_ids):
            image_ids = np.random.choice(image_ids, sample_size, replace=False)

        widths, heights, aspect_ratios = [], [], []
        brightness_vals, contrast_vals = [], []
        color_distributions = {'R': [], 'G': [], 'B': []}
        orientations = []
        file_sizes = []

        print(f"Analyzing {len(image_ids)} images...")

        for img_id in tqdm(image_ids):
            img_path = img_dir / f"{img_id}{ext}"
            if not img_path.exists():
                continue

            # 파일 크기
            file_sizes.append(img_path.stat().st_size / 1024)  # KB

            # 이미지 로드
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

            # 방향성 (가로 vs 세로)
            if w > h:
                orientations.append('landscape')
            elif h > w:
                orientations.append('portrait')
            else:
                orientations.append('square')

            # 밝기와 대비
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(gray.mean())
            contrast_vals.append(gray.std())

            # 색상 분포
            b, g, r = cv2.split(img)
            color_distributions['R'].append(r.mean())
            color_distributions['G'].append(g.mean())
            color_distributions['B'].append(b.mean())

        # 통계 출력
        print(f"\nImage Dimensions:")
        print(f"  Width:  {np.mean(widths):.1f} ± {np.std(widths):.1f} (min: {np.min(widths)}, max: {np.max(widths)})")
        print(f"  Height: {np.mean(heights):.1f} ± {np.std(heights):.1f} (min: {np.min(heights)}, max: {np.max(heights)})")
        print(f"\nAspect Ratio: {np.mean(aspect_ratios):.3f} ± {np.std(aspect_ratios):.3f}")

        orientation_counts = Counter(orientations)
        print(f"\nOrientation Distribution:")
        for orient, count in orientation_counts.items():
            print(f"  {orient:10s}: {count:4d} ({count/len(orientations)*100:.1f}%)")

        print(f"\nBrightness: {np.mean(brightness_vals):.1f} ± {np.std(brightness_vals):.1f}")
        print(f"Contrast:   {np.mean(contrast_vals):.1f} ± {np.std(contrast_vals):.1f}")

        print(f"\nColor Distribution (BGR):")
        print(f"  R: {np.mean(color_distributions['R']):.1f} ± {np.std(color_distributions['R']):.1f}")
        print(f"  G: {np.mean(color_distributions['G']):.1f} ± {np.std(color_distributions['G']):.1f}")
        print(f"  B: {np.mean(color_distributions['B']):.1f} ± {np.std(color_distributions['B']):.1f}")

        print(f"\nFile Size: {np.mean(file_sizes):.1f} ± {np.std(file_sizes):.1f} KB")

        # 결과 저장
        self.results[split]['dimensions'] = {
            'width': {'mean': float(np.mean(widths)), 'std': float(np.std(widths))},
            'height': {'mean': float(np.mean(heights)), 'std': float(np.std(heights))},
            'aspect_ratio': {'mean': float(np.mean(aspect_ratios)), 'std': float(np.std(aspect_ratios))}
        }
        self.results[split]['brightness'] = {'mean': float(np.mean(brightness_vals)), 'std': float(np.std(brightness_vals))}
        self.results[split]['contrast'] = {'mean': float(np.mean(contrast_vals)), 'std': float(np.std(contrast_vals))}
        self.results[split]['orientation'] = dict(orientation_counts)

        # 추천사항 생성
        if np.std(aspect_ratios) > 0.5:
            self.results['recommendations'].append(
                f"{split.upper()}: High aspect ratio variance - Use flexible resize strategies"
            )

        if np.std(brightness_vals) > 50:
            self.results['recommendations'].append(
                f"{split.upper()}: High brightness variance - Apply brightness augmentation"
            )

    def analyze_per_class_characteristics(self, samples_per_class=10):
        """클래스별 이미지 특성 분석"""
        print("\n" + "=" * 60)
        print("PER-CLASS CHARACTERISTICS ANALYSIS")
        print("=" * 60)

        class_stats = {}

        for class_id in range(17):
            class_name = self.meta_df[self.meta_df['target'] == class_id]['class_name'].values[0]
            class_images = self.train_df[self.train_df['target'] == class_id]['ID'].values

            if len(class_images) == 0:
                continue

            sample_ids = class_images[:min(samples_per_class, len(class_images))]

            widths, heights = [], []
            brightness_vals = []

            for img_id in sample_ids:
                img_path = self.train_dir / img_id  # img_id already includes .jpg
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_vals.append(gray.mean())

            if len(widths) > 0:
                class_stats[class_id] = {
                    'name': class_name,
                    'avg_width': np.mean(widths),
                    'avg_height': np.mean(heights),
                    'avg_brightness': np.mean(brightness_vals),
                    'samples_analyzed': len(widths)
                }

        # 출력
        print(f"\n{'Class':<5} {'Name':<50} {'Avg W x H':<15} {'Brightness':<12}")
        print("-" * 90)
        for class_id, stats in sorted(class_stats.items()):
            print(f"{class_id:<5} {stats['name']:<50} {stats['avg_width']:>6.0f}x{stats['avg_height']:<6.0f} {stats['avg_brightness']:>8.1f}")

        self.results['train']['per_class_stats'] = class_stats

    def compare_train_test(self):
        """훈련/테스트 데이터 분포 비교"""
        print("\n" + "=" * 60)
        print("TRAIN vs TEST COMPARISON")
        print("=" * 60)

        train_stats = self.results['train']
        test_stats = self.results['test']

        print("\nDimensions:")
        print(f"  Train - Width: {train_stats['dimensions']['width']['mean']:.1f}, Height: {train_stats['dimensions']['height']['mean']:.1f}")
        print(f"  Test  - Width: {test_stats['dimensions']['width']['mean']:.1f}, Height: {test_stats['dimensions']['height']['mean']:.1f}")

        print("\nBrightness:")
        print(f"  Train: {train_stats['brightness']['mean']:.1f} ± {train_stats['brightness']['std']:.1f}")
        print(f"  Test:  {test_stats['brightness']['mean']:.1f} ± {test_stats['brightness']['std']:.1f}")

        print("\nContrast:")
        print(f"  Train: {train_stats['contrast']['mean']:.1f} ± {train_stats['contrast']['std']:.1f}")
        print(f"  Test:  {test_stats['contrast']['mean']:.1f} ± {test_stats['contrast']['std']:.1f}")

        # 차이 분석
        brightness_diff = abs(train_stats['brightness']['mean'] - test_stats['brightness']['mean'])
        if brightness_diff > 20:
            self.results['recommendations'].append(
                f"DISTRIBUTION SHIFT: Brightness difference ({brightness_diff:.1f}) - Apply strong brightness augmentation"
            )

    def generate_augmentation_recommendations(self):
        """증강 전략 추천"""
        print("\n" + "=" * 60)
        print("AUGMENTATION STRATEGY RECOMMENDATIONS")
        print("=" * 60)

        print("\nBased on the analysis:")
        for i, rec in enumerate(self.results['recommendations'], 1):
            print(f"{i}. {rec}")

        # 문서 분류 특화 추천
        print("\nDocument-specific recommendations:")
        print("1. ROTATION: Documents can be slightly rotated (±7°) but avoid extreme rotations")
        print("2. PERSPECTIVE: Apply perspective transform (p=0.3-0.4) for camera angle variations")
        print("3. BRIGHTNESS/CONTRAST: Strong augmentation needed based on variance")
        print("4. GRID DISTORTION: Useful for scanner artifacts (p=0.3)")
        print("5. NOISE: Add Gaussian/Motion blur for print quality variations")
        print("6. COLOR: Minimal color jitter (documents are usually neutral)")
        print("7. AUGRAPHY: Document-specific degradations are valuable")

        # 구체적인 설정 추천
        train_stats = self.results['train']
        brightness_std = train_stats['brightness']['std']
        contrast_std = train_stats['contrast']['std']

        print("\nRecommended augmentation parameters:")
        aug_config = {
            'rotation_range': '±7 degrees',
            'brightness_limit': f'±{min(0.3, brightness_std/255):.2f}',
            'contrast_limit': f'±{min(0.3, contrast_std/255):.2f}',
            'perspective_prob': 0.35,
            'grid_distortion_prob': 0.35,
            'affine_translate': '±6%',
            'affine_scale': '(0.92, 1.08)',
            'mixup_prob': 0.3,
            'cutmix_prob': 0.0,
            'augraphy_prob': 0.4
        }

        for param, value in aug_config.items():
            print(f"  {param:25s}: {value}")

        self.results['augmentation_config'] = aug_config

    def save_results(self, output_path='eda_results.json'):
        """결과 저장"""
        output_file = Path('/root/upstage-cv-classification-cv-6/Seoyeon_Mun') / output_path
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def run_full_analysis(self):
        """전체 EDA 실행"""
        print("Starting comprehensive EDA analysis...")
        print("This may take several minutes...\n")

        # 1. 클래스 분포
        self.analyze_class_distribution()

        # 2. 훈련 데이터 이미지 속성
        self.analyze_image_properties(split='train', sample_size=500)

        # 3. 테스트 데이터 이미지 속성
        self.analyze_image_properties(split='test', sample_size=500)

        # 4. 클래스별 특성
        self.analyze_per_class_characteristics(samples_per_class=10)

        # 5. 훈련/테스트 비교
        self.compare_train_test()

        # 6. 증강 전략 추천
        self.generate_augmentation_recommendations()

        # 7. 결과 저장
        self.save_results()

        print("\n" + "=" * 60)
        print("EDA ANALYSIS COMPLETE")
        print("=" * 60)

if __name__ == '__main__':
    eda = DocumentEDA()
    eda.run_full_analysis()
