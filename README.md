# CV 6조
## Keep Going!!

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [문서연](https://github.com/UpstageAILab)             |            [최현](https://github.com/UpstageAILab)             |            [김수현](https://github.com/UpstageAILab)             |            [이승호](https://github.com/UpstageAILab)             |            [오정택](https://github.com/UpstageAILab)             |
|                            팀장, 실험총괄                             |                            ConvNext 모델링                             |                            ViT 모델링                             |                            실험 보조                             |                            실험 보조                             |

## 0. Overview
### Environment
<img width="322" height="188" alt="image" src="https://github.com/user-attachments/assets/97a3313b-36fe-4bc5-a77e-e75afa295b45" />


### Requirements
<img width="231" height="217" alt="image" src="https://github.com/user-attachments/assets/763e03cb-4beb-4ff1-87f4-bc31011a8c9c" />


## 1. Competiton Info

### Overview

이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용된 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

### Timeline

- 2025년 10월 31일 (금) 10:00 - Start Date
- 2025년 11월 12일 (수) 19:00 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

## 3. Data descrption

### Dataset overview

<img width="374" height="138" alt="image" src="https://github.com/user-attachments/assets/7a7860c7-13c2-410d-bc60-c42674d8b703" /\n>

학습 데이터셋 정보
train [폴더] 1570장의 이미지가 저장되어 있습니다.
train.csv [파일] 1570개의 행으로 이루어져 있습니다. train/ 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.
- ID: 학습 샘플의 파일명
- target: 학습 샘플의 정답 클래스 번호

meta.csv [파일] 17개의 행으로 이루어져 있습니다.
- target: 17개의 클래스 번호입니다.
- class_name: 클래스 번호에 대응하는 클래스 이름입니다.

평가 데이터셋 정보
test [폴더] 3140장의 이미지가 저장되어 있습니다.
- ID: 평가 샘플의 파일명이 저장되어 있습니다.
- target: 예측 결과가 입력될 컬럼입니다. 값이 전부 0으로 저장되어 있습니다.

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- https://www.kaggle.com/
- https://huggingface.co/timm
