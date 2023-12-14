---
title: High-Resolution Image Synthesis with Latent Diffusion Models
date: 2023-03-07 00:00:00 +09:00
categories: [Paper, Data Synthesis]
use_math: true
tags:
  [
    Computer Vision,
    Diffusion,
  ]
pin: true
---

[Capstone Design을 하면서 아이디어를 얻기 위해 논문을 읽었다. 아이디어를 얻기 위해 정리하는 것이다보니 주관적인 생각이 많다.]

[Latent Diffusion Models Github](https://github.com/CompVis/latent-diffusion)

- 해당 논문을 자세히 공부하게 된 이유(흥미로웠던 부분)
⇒대략적으로 살펴보았을 때, RGB에서 Depth map을 추정하고, 그 추정한 Depth map과 새로운 RGB image를 input으로 받아서, Depth map에 맞는 새로운 RGB image의 data가 들어가도록 하여 새로운 image를 만들어낸다.
- 이 것을 이용하면 우리는 real Thermal data와, 그 Thermal data와 매치(카메라의 거리나, 사람 수 등을 고려)되는 RGB image를 input으로 가져가면 RGB image에 맞는 Thermal Data를 생성할 수 있을 것 같다는 생각

# Abstract

![Untitled](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/9788721a-026c-4749-96c0-e84e9ac3022f)

- denoising autoencoders의 순차적인 적용을 통해 image formation(형성) 과정을 분해함으로써, diffusion models(DMs)은 image data 생성에서 최고의 성능을 얻을 수 있었고, 더 발전하고 있다.
- 추가적으로 그들의(diffusion models) 방법은 retraining없이 image generation process를 조절하기 위한 guiding mechanism을 가능하게 한다.
- 하지만, 이 model은 보통 pixel space에 직접 동작하므로, powerful DM의 최적화는 시간적 비용적으로 매우 비싸다.(순차적인 연산을 하기 때문에)
⇒DM의 quality와 flexibility를 유지하면서, 한정된 computational resoures로 DM training을 가능하게 하기 위해서, 논문은 powerful pretrained autoencoder의 latent(잠재하는) space에 DM을 적용했다.
- 이전의 연구와는 반대로, 앞서 말한 방법의 DM training은 complexity reduction(복잡성 축소)과 detatil preservation(세부사항 보존) 사이의 near-optimal point에 처음 도달했을 때를 고려하여, visual fidelity(시각적 정확도)를 크게 증가시킬 수 있다.
- model architecture에 cross-attention layer를 도입
⇒ Diffusion model을 text와 bounding box와 같은 일반적인 input을 적용할 수 있는 powerful and flexible generator로 만들었고, 높은 해상도의 synthesis(결과물, 합성물)이 convolutional 방식으로 가능해졌다.
- 논문에서 제안하는 Latent Diffusion Models(LDM)은 image inpainting(이미지 복원)과 class-conditional(class 조건부) image synthesis(합성)에서 최고의 성능을 얻었다.
- 또한 많은 task에서 높은 경쟁력있는 성능을 갖는다.
⇒unconditional image generation, text-to-image synthesis, super-resolution
(pixel-based DM에 비해 computation requirement를 상당히 줄이면서)
[* unconditional image generation은 특정한 input없이 새로운 image를 만들어 내는 것이다.]

# Introduction

- Image synthesis는 최근 가장 극적으로 발전한 Computer Vision분야 중 하나이다.
⇒하지만 가장 많은 연산량을 요구한다.(특히 high-resolution에서)
- GAN은 multimodal modeling으로 쉽게 확장할 수 없다.(multimodal의 복잡성으로 인해)
- 최근, denoising autoencoder의 체계로 만들어진 Diffusion model이 인상적인 결과를 보여주고 있다. ⇒ image synthesis와 그 외에 분야에서도

## Democratizing High-Resolution Image Synthesis

- Diffusion model은 likelihood-based model이다⇒ 연산량 많다.

likelihood function(=우도 함수): 가능성을 실제 측정하여 데이터로 나타내는 것

![Untitled 1](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d238208e-96c1-4647-b713-eeb7f53c0d27)

- Hight-Resolution Image Synthesis를 대중화(접근하기 쉽게 만들었다)
⇒ 매우 무거운 연산이 필요한데, 이것을 줄이기 위해 노력

## Departure to Latent Space

- 논문은 우선 이미 pixel space에서 train된 diffusion model을 분석하였다.

![Untitled 2](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/c9049d6d-2dfc-476b-ad3c-6248f4977119)

- Fig2는 rate-distortion(비율-왜곡) 균형을 보여준다.

[Perceptual Compression VS Semantic Compression]

두 방법 모두 데이터의 크기를 줄이는 방법(기본 정보는 유지하면서 불필요한 정보 제거)

하지만 데이터의 압축 방식에서 차이가 있다.

- Perceptual Compression: 인간의 인지에 중요하지 않은 정보를 제거하여 데이터양 줄이다.
ex) 이미지나 음성 압축에서 인간의 감각으로 인지하기 어려운 디테일을 줄인다.
- Semantic Compression: 중요한 Semantic(의미적인) 정보를 캡처
⇒데이터의 패턴과 규칙성을 식별하여 더 작은 symbol과 code로 나타낸다.

### Likelihood-based model는 대략적으로 2가지 stage로 학습을 한다.

- 1) high-frequency detail을 지우는 Perceptual Compression 단계
(semantic variation은 거의 학습하지 않는다)
- 2) 사실 상의 Generative model이 data에서 semantic and conceptual(의미의, 개념의) 요소들을 학습한다.(=Semantic Compression)

### 논문에서의 Training 방법

- 기존의 연구들을 따라 2가지 단계로 나눈다.

1) data space와 인지적으로 같은 lower-dimensional(낮은 차원의) representational space를 제공하는 Autoencoder를 학습⇒ 더 효율적이다.

* 중요한 것은, 이전의 연구들과는 반하여, 과도한 spatial compression(공간적 압축)에 의존하지 않아도 된다는 것이다.
⇒ 논문은 Diffusion model을 학습된 latent space(잠재공간)에서 학습시키기 때문
(공간 차원에 대하여 더 나은 scaling properties를 가능하게 한다.)
- 줄어든 복잡성(complexity)은 하나의 network pass만으로 latent space에서 효과적으로 image generation을 가능하게 한다.

### 논문은 최종적인 model을 Latent Diffusion Models(LDM)이라고 명명한다.

- 이러한 방법의 새로운 장점은 일반적인 autoencoding stage를 한번만 training시키면 되고, 이를 multiple DM training이나 가능하다면 완전히 새로운 분야를 탐구하는 것에 다시 사용할 수 있다.
⇒다양한 image-to-image, text-to-image task를 위한 diffusion model중 많은 수의 효과적인 탐구를 가능하게 한다.
- 후자의 경우, 논문은 transformer를 DM의 UNet backbone에 연결하고, 임의 유형의 token-based conditioning mechanism을 가능하게 한다.

## 논문에서의 기여

- 1) 순수한 transformaer-based 방법과는 다르게, 논문의 방법은 higher-dimentional data를 더 graceful하게 scaling하고, 그를 통해
(a) 이전 연구들보다 더 faithful and detailed(신뢰할 수 있고 자세한) reconstruction을 compression(압축) level에서 동작할 수 있다. ( Fig1 )

(b) megapixel(100만 픽셀) image의 high-resolution synthesis에 효과적으로 적용할 수 있다.

![Untitled 3](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/88c3c10d-20ab-4a57-ac74-e69d23656ad2)

- 2) computational cost를 상당히 줄이면서 다양한 task(unconditional image synthesis, inpainting, stochastic super-resolution)과 dataset에서 경쟁력있는 성능을 얻었다.
(또한 pixel-based diffusion 방법에 비해 inference(추론) cost를 상당히 줄일 수 있었다.)
- 3) encoder/decoder architecture와 score-based prior을 동시에 학습시켜야 하는 이전 연구들과 다르게, 논문의 방법은 reconstruction(복원)과 generative ability를 위한 delicate(섬세한) weighting이 필요하지 않다.
⇒논문의 방법은 매우 faithful한 reconstruction을 보장하고, latent space에서 매우 적은 regularization이 필요할 뿐이다.
- 4) 논문은 densely conditioned tasks(super-resolution, inpainting, semantic synthesis와 같은)를 위해, 논문의 model이 convolutional 방식으로 적용될 수 있고, 크게 만들 수 있다(~1024x1024 pixel의 image까지)
- 5) cross-attention 기반의 general-purpose(일반적으로 적용가능한) conditioning mechanism을 만들었다. ⇒multi-modal training이 가능하다.
(class-conditional, text-to-image, layout-to-image model을 학습시키는 것에 사용하였다.)
- 6) pretrained된 latent diffusion과 autoencoding model이 제공된다(github)
⇒논문은 Diffusion model training외의 다양한 task에 재사용할 수 있을 것이라고 예상한다.

# Method

![Untitled 4](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d2cff629-5308-48b5-ba9c-de3b00bff369)

- decoder 파트에 cross-modal counting model, 왜 thermal인가
- High-resolution image synthesis를 향한 diffusion models training에 필요한 computation을 낮추기 위해서, diffusion model이 perceptually(지각적으로) 연관성이 없는 detail들을 corresponding loss terms를 updersampling하여 무시하기는 하지만, DM은 여전히 pixel space에서 비용이 큰 function evaluation을 필요로 한다. (연산량이 매우 크고, 비용이 많이 든다)
- 논문은 이러한 문제를 피하기 위해 일반적인 learning phase(학습 단계)에서의 compressive과정을 명확하게 분리하는 방법을 제안한다.(Fig2)
- 이를 위해, 논문은 image space와 perceptually(지각적으로) 같은 space를 학습한 autoencoding model을 활용한다.
⇒하지만, computational complexity는 상당히 줄었다.
- 위의 방식들은 몇가지의 이점을 제공한다.

1) high-dimensional imaage space를 남김으로써, 계산적으로(computationally) 훨씬 효과적인 Diffusion Model을 얻었다
⇒ sampling이 low-dimensional space에서 수행되기 때문에

2) 논문은 UNet architecture에 상속된 Diffusion model의 inductive(귀납적인) bias를 이용한다.
⇒ spatial sturcture(공간적 구조, ex 1024x1024 image)와 같은 data에 대해 DM을 특히 효과적으로 만들 수 있다. 이를 통해 이전 연구의 방식에서 요구되었던 aggressive(공격적인), quality-reducing(질을 떨어뜨리는) compression level의 필요성을 줄일 수 있다.

3)마지막으로, general-purpose compression models을 얻을 수 있었다.
⇒ latent space가 다양한 generative model을 학습시키는 것에 사용될 수 있다.
⇒ 다른 Downstream application에 활용될 수 있다. ex) single-image CLIP-guided synthesis[25]

[예시 Paper "CLIPDraw: Exploring Text-to-Drawing Synthesis
through Language-Image Encoders
"](https://arxiv.org/pdf/2106.14843.pdf)

## 3.1. Perceptual Image Compression

- Perceptual Compression: 인간의 인지에 중요하지 않은 정보를 제거하여 데이터양 줄이다.
ex) 이미지나 음성 압축에서 인간의 감각으로 인지하기 어려운 디테일을 줄인다.
- perceptual loss와 patch-based adversarial objective의 결합에 의해 training된 autoencoder로 이루어져 있다.
⇒ local realism(지역적 사실성)을 강제함으로써 Reconstruction이 image manifold(이미지 다양체, image에 한에서만 다양하게 변화할 수 있다는 뜻으로 해석함)에 국한되도록 하고, 오로지 pixel-space loss(L2 또는 L1 objective)에만 의존하여 발생되는 bluriness를 피할 수 있다.

![Untitled 5](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/5808b1b7-3401-468e-be7d-18e4bae56163)

![Untitled 6](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/4ff00491-1a19-4608-9fcd-e8ddd77386a3)

![Untitled 7](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/e20c8dc1-4148-4ffc-b178-f8a7410f5275)

- 더 자세하게 말하자면, HxWx3의 RGB space를 갖는 image ‘x’가 주어졌을 때, encoder ‘E’는 x를 latent representaion(잠재적 표현)인 ‘z=E(x)’로 encoding한다.
⇒ 즉 input값을 z=E(x)로 encoding하여 latent representation으로 바꾼다.
- 그 후 decoder ‘D’는 latent representation으로부터 다시 image로 reconstructs한다.
⇒즉 latent z=E(x)값을 다시 D(z)=D(E(x))로 decoding하여 원래의 input x로 바꾼다.
(이 때 z=E(x)는 h x w x c의 크기)
- 중요한 것은 encoder가 factor f(f=H/h=W/w, 여기서 H,W는 input image의 spatial값이고, h, w는 output image의 spatial값이다.)를 통해 image를 downsampling하는 것이다.
(논문은 다른 downsampling factor로 f=2^m(m은 정수)을 연구하였다.)

![Untitled 8](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/8bd1186c-4528-45ef-9320-a2ba933802ff)

- 논문은 이 과정 이후의 Diffusion model이 learned latent space(학습을 통해 잠재력있는 정보만을 모아놓은 space, 즉 compression된 것이다) z=E(x)에 대한 two-dimensional(2가지 갈래를 의미하는 것 같다) structure에 동작할 수 있도록 하였다.
⇒ input x가 아닌 latent space(압축된 정보)를 Diffusion model에 적용하여 연산량이 줄었다.

## 3.2. Latent Diffusion Models

![Untitled 9](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/2ea94c3e-487f-445a-8835-570d20be5470)

![Untitled 10](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/cc1c164e-2f08-4a85-b4ca-632db2ac3686)

- Diffusion model은 probabilistic(개연론에 의거한) model이다.
⇒normally distrubuted variable을 점진적으로 denoising함으로써 data distribution ‘p(x)’를 학습하는 방식이다. (각 event의 확률이 이전 event에서 얻은 확률에 의존: Markov Chain)
- 논문에서 소개하는 model은 동일한 weighted sequence(순차)의 denoising autoencoder를 소개한다.
⇒Denoising Autoencoder $Eθ(xt, t); t = 1 . . . T$ (T는 sequence의 길이) (xt는 input x에 noise가 섞인 것)
→ denoising autoencoder Eθ는 xt에서 denoising된 것을 예측한다.

![Untitled 11](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/65753981-c5e0-40bb-90b8-5a1f8897a608)

위 공식은 Denoising Autoencoder의 결과와 Groundtruth사이의 Loss를 의미한다.

## 3.3. Conditioning Mechanisms

(가장 주요하게 생각했던 Point⇒ Depth-guided method에서 어떻게 적용되었는지 생각해보아야 한다.)

![Untitled 12](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/8d6a0f8b-b114-491a-9185-060e02f4ad5a)

- 다른 Generative model과 비슷하게, Diffusion model은 p(z|y)의 형태의 conditional distribution modeling을 하는 것에 원칙이 있다.

![Untitled 13](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/16da87b2-8f56-41fa-8772-ac8c24f8ce07)

- 위 식은 conditional denoising autoencoder를 나타낸 것 (Diffusion에서는 위 방식으로 conditional modeling을 가능하게 한다)
⇒ input ‘y’를 통해서 synthesis process를 통제한다.
(여기서 input y는 text, semantic map, 또는 image-to-image translation task 등을 의미)
**⇒ 우리는 input y를 thermal data로 넣는 방법에 대해 생각해보아야 한다.**

![Untitled 14](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/7198c9f9-c0c4-4457-b0fb-eb3d2e2b19db)

- 하지만, image synthesis의 context(맥락)에서, 다른 유형의 conditiong을 갖는 Diffusion model의 generative power를 합치는 것은 아직 탐구되지 않는 부분이 많다.

![Untitled 15](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/3bc87ca4-273a-4ff5-98ca-4f35171d7b05)

![Untitled 16](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/03fa78df-bd38-483a-9d98-a0b2e28d35ce)

- 논문은 Diffusion model을 더 flexible(유연한) conditional image generator로 만들었다.
⇒Cross-attention mechanism으로 근본적이 UNet backbone을 증대시킴으로써
- 논문은 다양한 modality로부터온 input ‘y’를 (y자체가 다양한 modality가 될 수 있다) pre-process하기 위해서, domain specific(분야에 특정적인) encoder ‘τθ’를 도입했다.
(domain specific encoder ‘τθ’는 modality input인 y를 중간 단계의 representation인 τθ(y)로 project한다.)
- τθ(y)는 cross-attention layer를 통해 UNet의 중간 layer에 mapping된다.
(attention 연산은 위에 공식으로 나와있다.)

![Untitled 17](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/38f954ce-bdc2-425c-aa26-f4e64b6404a7)

![Untitled 18](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d4550058-1557-45e6-be4b-35ed3dd818b6)

- $ϕi(zt)$는 Eθ(denoising autoencoder)를 위한 UNet의 (flattened, 편평해진) 중간 representation을 의미한다.
- W는 각각 Q, K, V의 학습 가능한 projection matrices(UNet에 적용하기 위한 Projection)

![Untitled 19](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/da3becc8-d071-4266-ba0a-8b8a4b3508ac)

- image-conditioning pair(image와 condition data)을 베이스로, conditional LDM(Latent Diffusion Model)을 학습시킨다.

![Untitled 20](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/9a999c9b-600b-4abe-aa27-d28d9b213c58)

- 위 식을 통해, τθ와 Eθ 모두 최적화(optimize)된다.
- 위 conditioning mechanism은 flexible하다. (τθ가 domain-specific 전문 시스템을 통해 parameter화될 수 있는 것처럼)