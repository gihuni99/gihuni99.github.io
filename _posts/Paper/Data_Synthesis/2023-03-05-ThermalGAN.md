---
title: ThermalGAN(Multimodal Color-to-Thermal)
date: 2023-03-05 00:00:00 +09:00
categories: [Paper, Data Synthesis]
tags:
  [
    Computer Vision,
    GAN,
    Data Synthesis
  ]
pin: true
---

[Capstone Design을 하면서 아이디어를 얻기 위해 논문을 읽었다.]

Image Translation for Person Re-Identification
in Multispectral Dataset

- 아래 논문(ThermalGAN)은 RGB를 GAN을 활용하여 Thermal로 변환하고, 이것을 ReID에 사용한다.
⇒ 그렇다면 우리는 RGB를 diffusion을 활용하여 Thermal로 변환하고, 이것을 Crowd Counting으로 이용하면 좋지 않을까?(그런데 이렇게 되면 image에 Thermal을 의존하게 되어, 어두운 곳에서 명확하게 알 수 있는 Thermal data의 장점을 이용하지 못할 것 같다)
→ Thermal를 RGB로 바꾸는 것이 더 큰 장점이 될 듯 하다.

[ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset](http://www.zefirus.org/articles/ee9462fb-befd-4679-9c26-acd551db8583/)

[https://github.com/vlkniaz/ThermalGAN](https://github.com/vlkniaz/ThermalGAN)

![Untitled](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/1145111a-3c8f-4446-8262-53dfbf0aeafd)

- ThermalGAN framework는 cross-modality color-thermal person ReID를 위해 제안되었다.
- 논문은 GAN을 이용해서 color image를 multimodal thermal probe set으로 변환하였다.
⇒Thermal 특징으로 Thermal histogram과 Feature descriptor를 사용하였다.
- ThermalWorld dataset을 사용

# Introduction

- 논문은 Person re-identification(ReID)를 위해서 ThermalGAN framework를 제안했다.
⇒ReID는 다양한 각도에서의 사람을 같은 사람으로 인식하는 것
- 우리는 ThermalGAN을 color-to-thermal image translation에 초점을 맞추려고 한다.

![Untitled 1](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/9372004b-38e0-4bdd-9781-3dc1189d633e)

[Paper "Infrared Colorization Using Deep Convolutional
Neural Networks"
](https://arxiv.org/pdf/1604.02245.pdf)

위 논문은 Near-infrared(근적외선,적외선 중 가시광선에 가까운 부분의 광선)을 RGB로 바꾸는 논문

- LongWave Infrared image(LWIR, 장파장 적외선)을 변환하는 것은 red channel과 thermal image사이의 낮은 correlation으로 아직 해결해야할 문제이다.

• **장파장 적외선**(Long-wavelength infrared): 파장 8~15µm, 주파수 20~37THz, 온도 -80~90°C. 열영상을 찍는 데 주로 사용되는 영역이다. 군용으로는 초음속으로 접근해오는 적 항공기 및 미사일 조기 탐지용으로도 많이 쓴다. 초음속 비행중 공기마찰에 의해 열이 나는데, 이때 이 장파장 적외선 대역의 적외선이 가장 많이 방출되기 때문.

![Untitled 2](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/cc7a5a96-f118-4cf8-8e05-6a504c823bbf)

[Paper "TV-GAN: Generative Adversarial Network Based Thermal to Visible Face
Recognition"](https://arxiv.org/pdf/1712.02514.pdf)

위는 thermal-to-image 논문(pix2pix이다, 얼굴을 복원해야 하는 특성상)
⇒ Crowd Counting이라면 pix2pix보다는 더 넓은 범위로 연산을 줄이는 것이 좋을 것 같다.

[https://github.com/junyanz/BicycleGAN](https://github.com/junyanz/BicycleGAN)

위는 image-to-image GAN인데, 잘 이용하면 Diffusion을 적용하여 Termal-to-image가 가능할 수 있을 것 같다.(Code도 있다.)

![Untitled 3](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/bb0d2273-88d1-4c62-9f77-d2451e449774)

**특이하게도 직접 Dataset을 만든 것 같은데, color-to-thermal translation을 위한 GAN training에 많은 data가 필요하여 만들었다고 한다. (이 dataset을 이용하면 반대의 경우인 thermal-to-color도 가능하지 않을까?)**

- pixel단위로 labeling되어 있다(10개의 class)
- **(ReID를 위해 다양한 각도에서 찍혔지만, 오히려 다양한 Dataset을 얻을 수 있을 듯하다)**
    
![Untitled 4](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/95dac2a0-2b5a-4469-9c20-052429293cfc)
    

(온도에 대한 정보도, 객체에 대한 상대온도로 표현되어 있다.)

![Untitled 5](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d22b9c80-2e2f-423f-b75e-ff690c44aa9e)

![Untitled 6](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/a0984439-2069-4816-8802-3bed243ba75c)

![Untitled 7](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/a37ae3ad-8ea1-4b8f-ae48-80f19711ecea)

- Color-to-thermal image translation은 아직 해결해야 할 문제이다
(하나의 color input에서 다양한 가능성의 thermal output이 나올 수 있기 때문)
Ex) 추운 가을과 더운 여름은 시각적으로는 같은 온도라고 느껴질 수 있음
하지만 피부의 온도는 다르다.
- 논문은 multi-modal image translation에서 최고의 성능을 보인 GAN framework들을 color-to-thermal task에 적용시켜보았다.
**(우리도 Diffusion모델(multi-modal image translation에 특화된) 여러개로 실험해볼 수 있다)**
- 논문은 GAN이 오차범위 5도씨의 범위로 물체의 온도를 예측할 수 있다는 것을 발견했다!
- 하지만 1도씨 이내의 오차범위를 가져야 한다.(옷과 피부 등을 구별하기 위해서는)
⇒우리는 굳이 옷과 피부까지 오차범위를 구할 필요가 있을까?
⇒5도씨의 오차면 충분할 수도 있다→ 모델을 더 가볍게 만들 수 있을 것 같다.
- 논문은 Translation quality를 올리기 위해 two-step approach를 이용
- 이때,  Relative Thermal Contrasts(눈, 눈썹과 같은)는 서로 다른 기상조건에 의한 평균 체온 변화에 거의 불변하다는 것을 발견
⇒가설을 세웠다.
(Relative Thermal Contrasts는 각 object에 대한 input color image와 평균 온도에 의해 좌우될 수 있다)
- 따라서 Object 온도 예측에 2가지 Step이 존재

![Untitled 8](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/aae9821a-706a-4dcb-9ac6-b481452219d4)

1) Input color image에서 평균 Object 온도를 예측
(논문에서는 이렇게 예측된 결과 image를 “thermal segmentation” image(S) 라고 칭한다.

2) Relative local temperature contrasts(비교적인 local 상대온도)(R)를 예측한다.

⇒Color image A와 Thermal Segmentation(S)의 영향을 받음

⇒ S와 R의 합이 최종적인 Thermal image(B)를 제공

![Untitled 9](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/b098f13e-36e3-4f17-b15b-ba8769bd5a92)

- 순차적인 thermal image 종합은 2가지 이점이 있다.
1) 해결해야할 문제가 첫번째 단계(generation of thermal segmentation)에만 있다.
2) 낮은 표준편차로 Thermal constrasts prediction(상대 온도 예측)의 질이 높아지고, 온도의 범위가 줄어든다.
- 논문은 color-to-thermal translation에서 multimodality를 다루기 위해, BicyleGAN framework를 사용하였다.
⇒ 하나의 color image에 대해 여러 color segmentation image를 합성하였다.
- 또한 Random noise sample대신, 훌륭한 background와 object 온도를 포함하는 temperature vector Ti를 사용한다.

![Untitled](ThermalGAN%20Multimodal%20Color-to-Thermal%20Image%20Trans%20a8968065e90341e58f0fab5f3e02a8ae/Untitled%2010.png)

- 이후의 내용은 수식과 함께 자세히 설명되어 읽는 것은 생략하였다.

# 종합

![Untitled 10](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/7f178877-60d9-4274-b226-6d111eadfee5)

- 종합하자면, T(Temperature vector)와 RGB image를 이용하여 S(thermal segmentation image)를 만든다.
    
![Untitled 11](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/3c68ae80-687e-4236-9524-f98bee2fc04f)
    
- 논문에서 말하는 thermal segmentation은 object별로 평균 온도를 예측
(segmentation과 유사하여 이름 붙인 듯 하다)
- 그리고 RGB와 thermal segmentation을 이용하여 상대적인 온도 R을 구한다.
(논문에서 언급하길 눈썹과 눈 같은 요소는 기상에 따른 변화가 적다→거의 일정하다)
⇒이를 이용하여 상대적인 온도를 나타내는 thermal data를 구한다.
- 최종적으로 R과 S를 합하여 thermal image를 구한다.

## 논문에서 흥미로웠던 점

- RGB image에서 기상에 의해 거의 변하지 않는 object를 이용하여 thermal data를 추정해내는 방법
⇒ 이때 GAN을 사용하였다.(우리는 diffusion을 사용하여 개선할 수 있을 것인지 생각해봄)
- 또한 해당 논문을 읽으면서 느꼈던 점은 온도정보를 추정할 수 있다는 것은 흥미로웠지만, Crowd Counting에서 Thermal data를 사용하는 이유는 어두운 환경에서 효과적이고, image에서 사람의 눈으로 보아도 식별이 어려운 사람을 효과적으로 나타낼 수 있는 점인데, image-to-thermal을 하여 Crowd counting을 한다면 thermal을 장점을 충분히 활용할 수 없을 것 같다는 생각을 했다.
⇒교수님께서 보내주신 stable-diffusion을 depth-to-image가 아닌 thermal-to-image로 적용할 수 있을 것 같다.
(stable-diffusion은 image에서 depth를 추정하여, 추정한 depth와 특정 image를 input으로 받고, 그 depth map에 맞도록 새로운 image를 만든다.

→ 그렇다면 thermal image와 thermal image에서 추정한 거리(거리에 따라 사람의 크기가 다르다)와 유사한 거리를 같는 image를 input으로 넣으면 thermal to image가 가능하지 않을까?