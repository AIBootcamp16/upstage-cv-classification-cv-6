# EDA 기반 증강 전략 (2025-11-05)

## EDA 핵심 발견사항

### 1. 밝기 분포 차이 (가장 중요!)
```
훈련 데이터: 평균 147.6 ± 36.2
테스트 데이터: 평균 174.3 ± 35.5
차이: 26.7 (약 18% 차이!)
```

**결론**: 테스트 데이터가 훈련 데이터보다 훨씬 밝음. 이는 모델이 훈련 중에 본 밝기 분포와 테스트에서 만나는 밝기 분포가 다르다는 의미로, **성능 저하의 주요 원인**일 수 있음.

### 2. 클래스별 밝기 차이
```
가장 어두운 클래스: car_dashboard (65.8)
가장 밝은 클래스: payment_confirmation (172.5)
차이: 106.7 (엄청난 차이!)
```

| 클래스 ID | 클래스 이름 | 평균 밝기 |
|-----------|-------------|-----------|
| 2 | car_dashboard | 65.8 |
| 5 | driver_license | 117.3 |
| 8 | national_id_card | 125.4 |
| 9 | passport | 137.5 |
| 11 | pharmaceutical_receipt | 146.0 |
| 0 | account_number | 149.5 |
| 15 | vehicle_registration_certificate | 156.3 |
| 16 | vehicle_registration_plate | 103.9 |
| 1 | application_for_payment... | 163.1 |
| 3 | confirmation_of_admission... | 164.9 |
| 4 | diagnosis | 164.6 |
| 7 | medical_outpatient_certificate | 166.2 |
| 12 | prescription | 164.8 |
| 13 | resume | 167.2 |
| 14 | statement_of_opinion | 168.5 |
| 6 | medical_bill_receipts | 169.9 |
| 10 | payment_confirmation | 172.5 |

### 3. 클래스 불균형
```
최소 샘플: 46개 (application_for_payment_of_pregnancy_medical_expenses)
최대 샘플: 100개 (대부분의 클래스)
불균형 비율: 2.17:1
```

### 4. 이미지 크기 및 방향
```
훈련 데이터:
  - 평균 크기: 499 x 536
  - 세로 방향: 64.2%
  - 가로 방향: 35.2%

테스트 데이터:
  - 평균 크기: 513 x 523
  - 세로 방향: 53.4%
  - 가로 방향: 46.0%

Aspect Ratio 표준편차: 0.31 (높은 변동성)
```

### 5. 대비(Contrast)
```
훈련: 47.1 ± 18.9
테스트: 47.8 ± 19.0
차이: 0.7 (거의 동일)
```

## 적용된 증강 전략 변경사항

### 주요 변경 1: 밝기 증강 강화 ⭐⭐⭐
**문제**: 훈련(147.6) vs 테스트(174.3) 밝기 차이 26.7
**해결**:
```python
# 변경 전
A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.4)

# 변경 후 (EDA 기반)
A.RandomBrightnessContrast(
    brightness_limit=0.25,  # 0.15 → 0.25 (67% 증가)
    contrast_limit=0.12,    # 0.2 → 0.12 (대비는 유사하므로 감소)
    p=0.6                   # 0.4 → 0.6 (적용 확률 50% 증가)
)
```

**효과**: 훈련 중 다양한 밝기의 이미지를 보게 하여, 테스트 데이터의 높은 밝기에 대응

### 주요 변경 2: Augraphy 밝기 범위 확대
```python
# 변경 전
Brightness(brightness_range=(0.9, 1.1), p=post_p)

# 변경 후
Brightness(brightness_range=(0.85, 1.25), p=post_p * 1.5)
```

**효과**: 문서 스캔/촬영 시 발생할 수 있는 다양한 조명 조건 시뮬레이션

### 주요 변경 3: Affine 변환 조정
```python
# 변경 전
A.Affine(translate_percent=0.04, scale=(0.94, 1.06), rotate=(-5, 5), p=0.4)

# 변경 후 (EDA 권장값 반영)
A.Affine(
    translate_percent=0.06,  # EDA 추천: ±6%
    scale=(0.92, 1.08),      # EDA 추천: (0.92, 1.08)
    rotate=(-7, 7),          # EDA 추천: ±7도
    p=0.5
)
```

**이유**: 문서는 스캔/촬영 시 약간의 회전/이동이 발생. Aspect ratio 변동성(std 0.31)을 반영

### 주요 변경 4: Perspective & Grid Distortion 확률 증가
```python
# 변경 전
A.Perspective(scale=(0.02, 0.05), p=0.25)
A.GridDistortion(distort_limit=0.1, p=0.25)

# 변경 후
A.Perspective(scale=(0.02, 0.05), p=0.35)  # EDA 추천
A.GridDistortion(distort_limit=0.15, p=0.35)  # EDA 추천
```

**이유**: 카메라 각도 변화 및 스캐너 왜곡을 더 자주 시뮬레이션

### 주요 변경 5: Mixup 파라미터 조정
```python
# 변경 전
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.3

# 변경 후
MIXUP_ALPHA = 0.3   # 클래스 불균형 완화
MIXUP_PROB = 0.35
```

**이유**: 클래스 불균형(2.17:1)을 완화하기 위해 Mixup 강도 및 확률 증가

### 주요 변경 6: Augraphy 강도 상향
```python
# 변경 전
AUGRAPHY_STRENGTH = 'light'

# 변경 후
AUGRAPHY_STRENGTH = 'medium'
```

**이유**: 실제 문서의 품질 변동성을 더 잘 반영

### 변경 7: 색상 지터 감소
```python
# 변경 전
A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3)

# 변경 후
A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01, p=0.2)
```

**이유**: EDA에서 RGB 값이 매우 유사함 (R:145.9, G:148.2, B:149.3). 문서는 대부분 중립 색상

## 기대 효과

### 1. 밝기 분포 차이 해결 (가장 중요!)
- 훈련 중 더 넓은 밝기 범위의 이미지를 경험
- 테스트 데이터의 높은 밝기(174.3)에 대한 일반화 성능 향상
- **예상 개선폭: 3-5% F1 점수 향상**

### 2. 클래스별 밝기 차이 대응
- Car dashboard(65.8)부터 medical docs(170+)까지 다양한 밝기 학습
- 모든 클래스에 대해 균일한 성능 달성

### 3. 클래스 불균형 완화
- Mixup 강화로 적은 샘플 클래스(46개)의 효과적인 데이터 증대
- Class weights와 함께 사용하여 시너지 효과

### 4. 기하학적 변형 강화
- 다양한 카메라 각도, 스캔 품질에 강건한 모델
- Aspect ratio 변동성에 대응

## 실행 방법

```bash
cd /root/upstage-cv-classification-cv-6/Seoyeon_Mun
python main2.py
```

## 모니터링 포인트

1. **Validation F1 범위**: 0.88-0.92 예상 (이전: 0.91-0.94)
   - 약간 낮아질 수 있지만, 이는 과적합 감소로 인한 **건강한 신호**

2. **Fold 간 일관성**: 표준편차가 감소해야 함
   - 이전: fold 3/5가 특히 낮았음 (0.71, 0.72)
   - 목표: 모든 fold에서 0.83+ 달성

3. **Holdout F1**: 최종 성능 지표
   - 목표: 0.86+

4. **리더보드 점수**: 가장 중요한 지표
   - 밝기 분포 차이 해결로 **실질적인 성능 향상** 기대

## EDA 결과 파일

- EDA 스크립트: `eda_analysis.py`
- EDA 출력 로그: `eda_output.txt`
- EDA 결과 JSON: `eda_results.json`

## 참고사항

- 밝기 증강 강화로 인해 초반 epoch의 학습이 약간 느려질 수 있음
- 이는 정상이며, 더 강건한 특징을 학습하고 있다는 증거
- 최종 성능(리더보드)이 가장 중요한 지표
