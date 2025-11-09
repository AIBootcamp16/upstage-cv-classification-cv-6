# 과적합 해결을 위한 개선 사항

## 문제 진단
- Validation F1이 너무 높음 (0.94-0.98) → 과적합 의심
- 최근 실험에서 성능 저하 (0.9664 → 0.9384)
- Fold 간 성능 차이가 큼 (특히 Fold 5)

## 적용한 해결책

### 1. 정규화 강화
**변경 전:**
- Dropout: 0.5 (너무 높음)
- Drop Path: 없음
- Weight Decay: 0.05
- Label Smoothing: 0.02 (너무 낮음)

**변경 후:**
- Dropout: 0.3 (적절한 수준)
- **Drop Path: 0.2 추가** (Stochastic Depth)
- Weight Decay: 0.1 (증가)
- Label Smoothing: 0.1 (5배 증가)

### 2. 데이터 증강 강화
**변경 전:**
- Mixup/CutMix: 비활성화
- 증강 확률: 낮음 (0.15-0.4)
- 증강 강도: 약함

**변경 후:**
- **Mixup 활성화**: prob=0.5, alpha=0.4
- **CutMix 활성화**: prob=0.5, alpha=1.0
- 증강 확률 증가: 0.25-0.6
- 증강 강도 증가:
  - Affine: translate 0.04→0.06, scale (0.94,1.06)→(0.92,1.08), rotate ±5→±7
  - Perspective: p=0.25→0.35
  - GridDistortion: p=0.25→0.35, distort 0.1→0.15
  - Brightness/Contrast 강도 증가
  - MotionBlur/MedianBlur 추가

### 3. EMA (Exponential Moving Average) 추가
- **새로운 기능**: 모델 파라미터의 이동 평균 사용
- Decay: 0.9995
- 효과: 더 안정적이고 일반화된 모델

### 4. 학습 설정 조정
**변경 전:**
- Batch Size: 12 (효과적: 48)
- Epochs: 20
- Learning Rate: 3e-4
- Patience: 5
- Hold-out: 비활성화

**변경 후:**
- Batch Size: 16 (효과적: 48 유지)
- Epochs: 30 (증가)
- Learning Rate: 2e-4 (감소)
- Patience: 7 (증가)
- **Hold-out: 활성화** (10%)

## 기대 효과

1. **일반화 성능 향상**: 과적합 감소로 테스트 성능 개선
2. **안정성 향상**: EMA를 통한 더 안정적인 예측
3. **강건성 향상**: 강력한 증강으로 다양한 입력에 대응
4. **검증 신뢰도 향상**: Hold-out set으로 최종 성능 확인

## 주의사항

- Validation F1이 0.85-0.90 정도로 내려갈 수 있지만, 이것이 더 건강한 상태
- 학습 시간이 증가함 (30 epochs, 강한 증강)
- Hold-out F1을 최종 성능 지표로 활용

## 실행 방법

```bash
cd /root/upstage-cv-classification-cv-6/Seoyeon_Mun
python main1.py
```

## 추가 튜닝 옵션

더 과적합이 심할 경우:
1. Mixup/CutMix 확률을 0.7로 증가
2. Drop Path를 0.3으로 증가
3. Weight Decay를 0.15로 증가
4. 더 강한 증강 적용

성능이 너무 낮아질 경우:
1. Mixup/CutMix 확률을 0.3으로 감소
2. Drop Path를 0.1로 감소
3. Label Smoothing을 0.05로 감소
