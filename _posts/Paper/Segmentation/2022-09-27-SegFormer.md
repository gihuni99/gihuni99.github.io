---
title: SegFormer 논문 리뷰
date: 2022-09-27 00:00:00 +09:00
categories: [Paper, Segmentation]
use_math: true
tags:
  [
    Computer Vision,
    Paper,
    Semantic Segmentation
  ]
pin: true
---

## 특징

- Multi-Scale Feature를 output으로 하는 계층적인 구조의 Transformer Encoder사용

⇒ ViT와 다르게 Positional Encoding을 사용하지 않는다.

- 복잡한 Decoder를 사용하지 않고, MLP로만 이루어진 MLP decoder를 사용

⇒ Encoder에서 얻은 Multi-Scale Feature를 결합한다.

⇒ 각 Feature Map에서의 Local Attention+합쳐진 Feature Map에서의 Global Attention

⇒ 강력한 Repressentation을 얻는다.

# 1. Introduction

## 1.1. NLP에서 사용되었던 Transformer를 Vision Task에 사용하면 발생되는 문제점

- ViT의 output은 Single-Scale의 낮은 Resolution Feature를 갖는다.
- Large Image에 대해 많은 계산량이 요구된다.

## 1.2. SegFormer의 Encoder & Decoder

- Positional Encoding을 사용하지 않고, Hierachical Transformer Encoder를 사용한다.

⇒Positional Encoding은 Train/Test시 Image의 Resolution이 다르면 성능이 저하되는 특징이 있다. Interpolation이 발생하기 때문인데, Hierachical Transformer Encoder는 Train Image와 다른 해상도를 가지는 Test Image에 대해 Interpolation이 발생하지 않는다.

- 계산량을 줄인 간단한 All-MLP Decoder

# 2. Method

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b844cb46-d80b-4059-8321-c7cbd69d27ed)

## SegFormer의 구성

- Hierarchical Transformer Encoder

⇒ High-Resolution에서 Coarse Feature를, Low-Resolution에서 Fine Feature를 생성하기 위해 Hierarchical(계층적인)한 Transformer Encoder를 사용한다.

- All-MLP Decoder

⇒ 마지막 최종 Semantic Segmentation Mask를 얻기 위한 Multi-Level Feature를 합치기 위해 All-MLP Decoder를 사용한다.

## 1) Hierarchical Transformer Encoder

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/00b84722-91fa-4e2d-a49d-300a4e9f45c2)

- Hierarchical Transformer Encoder는 Mix Transformer Encoder(MiT)라고도 하며 ViT에서 영감을 받아 Semantic Segmentation에 적용시켰음을 의미한다.

### 1-1) Hierarchical Feature Representation

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/e9a4c211-3f83-4be3-9cc8-69a3ca3d6465)

- Single-Resolution Feature Map을 생성하는 ViT와는 다르게 SegFormer은 High-Resolution Coarse Feature와 Low-Resolution Fine-Grained Feature를 가지는 Multi-Level Feature Map을 통해 Semantic Segmentation 성능을 향상시켰다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a809b514-ab0a-4466-a968-f2be13406d82)

- 각 Stage(총 4-stage)별로 Feature Map은 위 식과 같은 Resolution과 Channel을 갖는다.

### 1-2) Efficient Self-Attention

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c3ab8be1-7d7f-4bd2-824b-b46ef3c826c9)

- Encoder의 Self-Attention Layer는 많은 연산량을 요구한다.

⇒ SegFormer은 Patch Size가 16x16이 아닌 4x4이므로 더 많은 Parameter값들을 계산해야 한다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/dd33d246-938b-4f95-91b6-852c7d8e2342)

- 기존의 Multi-Head Self-Attention은 Q, K, V를 모두 N(HxW)xC차원을 갖는 행렬로 만들어 위와 같이 계산

⇒ O(N^2)의 계산 복잡도 → Large Image가 들어오면 Model이 급격하게 무거워짐

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bc3f6418-8d8d-40ee-aed1-932776d446c7)

- R(Reduction Ratio)을 사전에 정의하여 N(HxW)채널을 줄이는 Sequence Reductio Process 적용

**⇒ Reshape(N/R, C*R)(K) : NxC →(N/R)x(C*R)차원으로 변경**

**⇒ Linear(C*R,C) : C*R→ C차원으로 변경**

(R은 실험을 통해 [64, 16, 4, 1]로 설정)

**⇒ 계산복잡도 :  O(N^2)→O((N^2)/R)**

### 1-3) Overlapped Patch Merging

- ViT에서는 NxNx3 Patch를 1x1xC 벡터로 표현하였다.

⇒ 각 Patch들은 Non-Overlap된 상태이다. 따라서 각 패치 간의 Local Continuity(순서)보존 어렵다.

- **Overlapped Patch Merging으로 해결**

단순히 4x4 Patch Size로 나누어 Vector Embedding을 하는 것이 아니다.

⇒ **K(Patch Size 또는 Kernel Size), S(Stride), P(Padding Size)를 사전 정의**

⇒ B(Batch)xC(Channelx(Stride^2))xN(Number of Patch)의 차원으로 Patch를 분할

⇒ C(Embedding dimention)xW(Width)xH(Height)의 차원으로 Merging 수행

(CNN이 Sliding Window로 조금씩 겹치며 연산하는 것과 비슷한 방식)

(2x2xCi→1x1xCi+1)⇒(H/4xW/4xC1→H/8xW/8xC2)

### 1-4) Positional-Encoding Free Design

- ViT에서는 Local Information을 주기 위해 Positional Encoding을 추가

⇒ Input Resolution이 고정되어 있어야하고, 다르다면 Interpolation을 통해 크기를 맞춰야 하기 때문에 성능이 저하될 수 있다.

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7c58327b-8702-447f-995b-23fa326876de)

Positional Encoding대신 위 식과 같이 3x3 Convolution(Stride=1/Padding=1)을 FFN에 적용하였다.

⇒ Zero Padding을 통해 leak location 정보를 고려할 수 있다.

(+ Parameter 수를 줄이기 위해 3x3 Convolution을 Depth-Wise Convolution으로 사용)

## 2) Lightweight ALL-MLP Decoder

- MLP layer로만 구성된 Decoder이다.

⇒ 기존 다른 Model의 Decoder와 다르게 수작업 및 연산량이 감소

- Hierarchical Transformer Encoder에서 Larger Effective Field를 가졌기 때문에 간단한 Decoder가 설계 가능했다.

### 2-1) MLP Decoder의 연산 순서

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/83462fb0-0e50-4c2e-9746-b32650d53338)

1. Multi-Level Feature들의 Channel을 동일하게 설정
2. Feature Size를 Original Image의 1/4크기로 동일하게 설정
3. Feature들을 concat하고 이 과정에서 4배로 증가한 Channel을 다시 복구
4. 최종 Segmentation Mask를 예측