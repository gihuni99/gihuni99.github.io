---
title: Multimodal recurrent neural networks
date: 2023-01-15 00:00:00 +09:00
categories: [Paper, RGB-D Segmentation]
tags:
  [
    Computer Vision,
    Paper,
    RGB-D,
    Segmentation
  ]
pin: true
---

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2c382f52-0cf9-40e5-b91a-b7527b913a3d)

- 논문은 Multimodal Recurrent Neural Netwoks(RNNs)라는 RGB-D scene Semantic Segmentation을 위해 새로운 방법을 제안한다.
- 이 Network는 RGB color channels와 Depth maps가 Input Source로 주어지면 Image pixel을 분류하는 것에 최적화되어있습니다.
- 이 Network는 동시에 Information Transfer layers를 통해 교차로 연결된 2개의 RNN의 Training을 수행
⇒ Information Transfer Layer는 관련있는 Cross-Modality Features를 적응적으로 추출하기 위해 학습되어 있다.
- 각 RNN 모델은 각각 자신의 이전 Hidden State에서부터 나온 Representation(표현)과 다른 RNN의 이전 Hidden State에서 온 Transferred Pattern(전송된 패턴)을 학습한다.
⇒ 이를 통해 Model-Specific과 Cross-Modality Feature모두가 유지
- 논문은 Quad-directional (4방향) 2D-RNN의 구조를 사용한다
⇒ 2D image의 short- and long-range contextual information(단, 장거리 문맥 정보)를 modeling하기 위해서

# Introduce

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4db0d0f9-ba65-4351-9c94-a3994c581017)

- 위는 Depth information이 RGB image에서 유사한 모양을 구별하는데 어떻게 도움주는지 예시
- Scene Image가 messy(지저분한) object를 여러 개 포함하고 있는 경향
⇒ Scene Labeling(Semantic Segmentation)이 어렵다
⇒Messy Objects는 그들의 외형과 형상에 미치는 요인들 때문에 변형될 수 있다.

⇒ 각 modality 내부의 각 Pixel에 대한 Neighborhood/Contextual information을 활용

- 일반적으로 pixel의 Feature Representation은 target pixel(대상 픽셀)을 포함하는 Local Patch(Scene Image에서 잘라낸)에서 추출되고 Classification에 이용된다.
- Long-range/Global Contextual Information(Distant image patch)도 Local Pixel classification을 위해서 중요하다.
- 하지만 Local and Global Contextual Information 모두 적절하게 활용되어야 한다.
⇒Pixel Feature Representation의  Discriminative(식별할 수 있는) Features와 abstract(추상적인)/top-level Features를 적절한 균형을 유지하기 위해서

---

- RNN은 Contextual Information을 Local Feature Representation으로 Encoding하는 것에 매우 성공적이 결과 보임
⇒ Recurrent Model은 Feedback connection이 있어서 현재 State가 미래 state의 계산에 관여한다.
⇒RNNs이 long and short-range dependency(종속성)을 modeling해야 하는 음성 인식과 MLP(자연어 처리)에 효과적으로 사용됨
- 논문에서는 RNN을 각 modality내에서 Contextual Information을 modeling하는 것에 사용
⇒ 하지만, 일반적으로 RNN은 단일 modality signal에서만 사용
- 논문에서는 Contextual Information을 Multimodal RGB-D data의 Local Repredentations으로 modeling하는 새로운 Multimoal RNNs method를 소개
⇒ 먼저 Local RGB-D image patch(RGB image와 Depth plane에서의)로부터 Feature extraction을 위해 CNN을 학습시킨다.
- 위 Convolutional Local Feature가 논문의 multimoal RNNs의 input을 형성한다
⇒추가적으로 contextualize(맥락과 관련짓다)하고 modality전체에서 유용한 pattern을 추출하기 위해서

⇒ 논문의 Model은 더 많은 modality를 고려하여 Prediction Task를 수행할 수 있도록 쉽게 확장 가능하다.

---

- 논문의 새로운 Multimodal RNNs method는 basic Quad-directional 2D-RNNs Structure을 기반으로 구축되었다.
- Quad-directional 2D-RNN은 4개의 hidden State를 포함한다
⇒각 hidden state는 4개의 가능한 방향의 특정 2D direction으로 image를 순회하는 것에 전념
(top-left, top-right, bottom-left and bottom-right)
- RGB image와 Depth plane, 2개의 modality를 처리하기 위해, 논문의 model은 2개의 RNNs
⇒그 중 하나는 하나의 input modality의 representation을 학습하기 위해 할당된다
- 두 개의 RNN modality를 연결하고 Information Fusion을 위해서, 논문은 RNN을 교차하여 연결하는 information transfer layer를 개발

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c2d36680-4447-4abd-baca-ae838c94bc5b)

- Transfer layer는 하나의 modality에서 다른 modality로 relevant(관련된) pattern을 선택하고 전송하는 것에 대한 학습을 책임진다.
- 구체적으로, input image에 있는 모든 patch에 대해, 그리고 하나의 modality에 대한 RNN hidden representation을 학습하는 과정에서, 논문의 방법은 각각의 modality내의 Contextual information만을 Encoding할 뿐만 아니라, 다른 modality에서 관련 Contextual pattern을 Encoding하는 방법도 학습한다.
- 결과적으로, 논문의 방법은 local pixel representation을 위한 강력한 context-aware and multimodal feature를 학습할 수 있다

---

- 논문의 방법은 기존의 Deep multimodal learning method와 다르다
⇒기존의 방법들은 일반적으로 시작 지점에서 input을 Concat하거나, middle level로 학습된 Feature를 Concat한다.
→ high-level common(공통) feature를 multimodal data의 Representation으로 추출하기 위해서
⇒ 위 방법은 서로 다른 modality간의 공통된 pattern을 찾는 것에 중점을 둔다
⇒ 공통된 pattern은 extract에 중요하지만, 이 방법은 RGB channel 내부의 texture pattern과 같이, single modality내에서 매우 차별적인 중요한 modality specific information을 놓치기 쉽다.
- 구체적으로, 논문의 model은 각 modality의 Feature를 학습하도록 RNN model을 할당함으로써 Modality-Specific information을 유지한다.
- 또한, 논문의 model은 Information Transfer Layers를 사용함으로써 관련 있는 cross-modality pattern을 적응적으로 전송하여, modality사이에 정보 교환을 가능하게 한다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/321b4f73-a0ec-4a28-b98a-321b73eec5cc)

- 논문은 RGB-D scene semantic segmentation을 위한 새로운 방법을 제시
- 2개의 quad-directional 2D-RNN사이의 Information transfer layer를 도입
- Transfer layer는 modality전체에서 관련 contextual information을 추출하고, 각 modality가 공유된 정보를 capture할 수 있는 context-aware feature를 학습할 수 있도록 돕는다.