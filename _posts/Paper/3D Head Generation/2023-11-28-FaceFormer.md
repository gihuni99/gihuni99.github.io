---
title: FaceFormer:Speech-Driven 3D Facial Animation with Transformers 논문 리뷰
date: 2023-11-28 00:00:00 +09:00
categories: [Paper, 3D Head Generation]
use_math: true
tags:
  [
    Paper,
    3D Head Generation
  ]
pin: true
---

# Abstract

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ddf679cc-7f0a-494c-a32d-dcf89af1cda8)

- “Speech-driven 3D facial animation”은 human faces의 복잡한 geometry와 3D audio-visual data의 한정된 가용성으로 인해 challenge하다.
- 이전 연구들은 보통 한정된 context를 통해 짧은 audio window(순간)들의 phoneme(음소: 소리의 가장 짧은 단위, ex. /p/)-level feature를 학습하는 것에 주력했다.
⇒ 때때로 부정확한 lip movements를 보인다.
- 이러한 한계를 해결하기 위해, 논문은 **“Transformer-based autoregressive model(FaceFormer)”**를 제안한다.
⇒ **long-term audio context를 encoding하고 autoregressive하게 3D face mesh animation의 sequence를 예측**한다.
- **Data 부족 문제를 해결**하기 위해, **“self-supervised pre-trained speech representations”**를 사용하였다.
- 또한, 이 **task에 적합한 2개의 “biased attention mechanism”을 고안**하였다.
⇒ **“biased cross-modal multi-head(MH) attention”, “biased causal multi-head(MH) self-attention” with periodic positional encoding strategy**
- **전자(biased cross-modal MH attention)는 audio-motion modalities를 효과적으로 align**한다.
반면, **후자(biased causal MH self-attention)는 더 긴 audio sequence를 generalize**할 수 있게 한다.

---

***Figure 1***

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b32b4428-32d5-465f-8f98-5d1376e8aa52)

- Figure 1은 FaceFormer의 대략적인 diagram을 보여준다.
- **raw audio input과 neutral 3D face mesh**가 주어졌을 때, 논문은 **end-to-end Transformer-based architecture**를 제안한다.
- FaceFormer은 **autoregressive하게 정확한 lip movements를 갖는 realistic 3D facial motion sequence를 synthesis**할 수 있다.

---

# 1. Introduction

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6c132284-4b03-4a47-9460-76c6e69e5cec)

- Speech-driven 3D facial animation은 학계와 산업계 모두 매력적인 연구 분야가 되어왔다.
⇒ virtual reality, film production, games, education 등 광범위한 분야에 잠재적으로 이롭다.
- Realistic speech-driven 3D facial animation은 임의의 speech signal을 통해 3D avatar의 vivid(생생한) facial expressions를 자동적으로 animate하는 것에 목표를 둔다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/01c8ef19-8d95-4db7-a340-7617aadaeaa2)

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/72a00196-e9d1-4455-b732-fe3da0f52a30)

- 논문은 **2D pixel값에 대한 animating(eg. photorealistic talking-head animation)보다 3D geometry를 animating하는 것에 주력**했다.
- 대부분의 기존 연구들이 방대한 2D video datasets가 주어졌을때, talking heads의 2D videos를 만드는 것에 목표를 두었다.
- 하지만, **생성된 2D videos는 3D 공간에서 3D model을 animate해야 하는 3D games나 VR같은 분야에 직접적으로 적용할 수 없다.**
- 몇몇의 method들은 **3D facial parameters를 획득하는 것에 2D monocular videos를 이용**한다.
⇒ **신뢰할 수 없는 결과**를 낼 수도 있다.
→ **synthetic 3D data의 quality는 3D reconstruction techniques의 정확도에 의해 한계가 존재**
(**3D에서의 미묘한 변화를 capture할 수 없다.**)
- **speech-driven 3D facial animation에서, 대부분의 3D mesh-based works는 짧은 audio windows를 input으로 사용**한다.
⇒ **facial expression의 변화를 모호하게 나타낼 수도 있다.**
- “Karras에 의해 주목 되었듯이, **전체 face를 사실적으로 animate하기 위해서는 “longer-term audio context”가 필요**하다.
- “MeshTalk”는 audio sequence를 modeling함으로써 longer audio context를 고려했지만, Mel spectral audio features로 model을 training하는 것은 부족한 data setting으로 정확한 lip motions를 synthesis하는 것에 실패했다.
- 3D motion capture data를 수집하는 것 또한 비용이 비싸다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0d17d29c-382e-40eb-a2ca-20b06c0a9c52)

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c7ea2274-9e9a-4cea-8758-1d66426a9deb)

- 위와 같은 **“long-term context”와 “3D audio-visual data부족” 문제를 해결하기 위해, 논문은 “transformer-based autoregressive model”(Fig 1)을 제안**한다.
**⇒ (1) 전체 얼굴의 매우 사실적인 animation을 가능하게 하기 위해 “longer-term audio contest”를 capture한다.
⇒ (2) data부족 문제를 해결하기 위해 “self-supervised pre-trained speech representations”를 효과적으로 사용한다.
⇒ (3) 일시적으로 안정된 facial animation을 만들기 위한 face motions의 흐름(이전상황까지)을 고려한다.**

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8c671f49-8d06-4dfe-9983-c2e0587fefef)

- “Transformer”는 NLP(Natural Language Processing)와 Computer Vision분야에서 놀라운 성능을 보였다.
- “LSTM”과 같은 **sequential model은 longer-term context를 효과적으로 학습하는 것을 저해하는 “bottleneck”을 가지고 있다.**
- RNN-based models와 비교해보았을 때, **transformer는 오로지 attention mechanisms을 기반으로 하여 long-range context dependencies를 더 잘 capture**할 수 있다.
- 최근, transformer는 body motion synthesis와 dance generation분야의 발전에 영향을 미치기도 하였다.
- transformer의 성공은 **self attention mechanism을 결합시킨 transformer의 design의 덕분**이다.
⇒**representation의 모든 부분을 명확하게 처리함으로써 short-range relation과 long-range relation을 모두 효과적으로 modeling할 수 있음**

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/792312c5-9f45-4708-bc77-fbedda7901d7)

- **audio sequences에 vanilla transformer architecture를 직접적으로 적용하는 것은 “speech-driven 3D facial animation” task에서 잘 동작하지 않는다.** 따라서 논문은 이러한 문제를 해결해야 했다.
- **(1) Transformer는 본질적으로 데이터가 매우 많이 필요**하다.(training의 상당한 large dataset이 필요) 제**한된 3D audio-visual data의 가용성을 고려하여, 논문은 self-supervised pre-trained speech model(wav2vec 2.0)의 사용에 대해 연구**하였다. **“wav2vec 2.0”은 unlabeled speech의 large-scale corpus(말뭉치)로 학습되기 때문에, 풍부한 phoneme 정보를 학습**하였다.
⇒ **한정된 3D audio-visual data가 충분한 phonemes를 포함하지 못할 수도 있지만, 논문은 pre-trained speech representations가 data가 부족한 상황에서 speech-driven 3D facial animation task에 도움이 될 것이라고 예상**하였다.
- **(2) transformer의 고정된 encoder-decoder attention은 modality alignment를 수행할 수 없다. 따라서 논문은 “audio-motion alignment”를 위한 “alignment bias”를 추가**하였다.
- **(3) 논문은 speech와 face motions 사이의 correlation을 modeling하는 것은 “long-term audio context dependency”를 고려하는 것이 필요하다고 주장**한다. 따라서, **encoder의 self-attention에 대한 attention scope를 제한하지 않았고, 그 결과 encoder가 long-range audio context dependency를 capture**할 수 있게 되었다.
- **(4) “sinusoidal position encoding을 하는 transformer는 training동안 사용했던 것보다 sequence lengths가 긴 것들에 대해서 generalization 성능이 약하다.**
(**position encoding(PE)**: image를 token단위로 변환할 때, 위치정보를 유지하기 위해 사용하는 encoding방법, sinusoidal position encoding은 PE가 sine형태인 것이다. 자세한 것은 추후에 알아보자)
- “**ALiBi(Attention with Linear Biases)”에서 영감**을 받아, 논문은 **query-key attention score에 “temporal bias”를 추가하였고, longer audio sequences에 대한 model의 generalization성능을 향상시키기 위해 “periodic positional encoding” 하는 방법을 구상**하였다.
(ALiBi는 word embedding에 positional embedding 값을 추가하지 않고, query-key attention score에서 각 distance에 따라 score에 penalty를 주는 방식, positional encoding의 한 방법이다)

### [Main Contributions]

#### An autoregressive transformer-based architecture for speech-driven 3D facial animation.

***speech-driven 3D facial animation을 위한 “autoregressive transformer-based architecture”***

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ecc2b349-d28d-43df-8a8b-8bec28604a97)

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6177db68-0922-43df-9983-1d605d552717)

- FaceFormer는 “long-term audio context”와 “face motions의 history”를 encoding한다.
⇒ autoregressive하게 animated 3D face meshes sequence를 예측하기 위함
- 이를 통해 매우 사실적이고, temporally(시간적으로, 흐름이 자연스럽다는 이야기인 것 같다) 안정적인 whole face animation을 구현할 수 있다.

#### The biased attention modules and a periodic position encoding strategy

***“baised attention module”과 “periodic position encoding” 전략***

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7576def1-2d5c-4ab3-bd6f-3cac9b7aa4f2)

- 논문은 다음과 같은 구조를 면밀하게 구성하였다.
**1) biased cross-modal MH attention**
⇒ 서로 다른 modality를 alignment하기 위함
**2) biased causal(인과관계의) MH self-attention with periodic position encoding**
⇒ longer audio sequences에 대한 generalization을 향상시키기 위함

#### Effective utilization of the self-supervised pretrained speech model

***self-supervised pre-trained speech model의 효과적인 사용***

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a23ce260-ced7-44d7-b591-fd658eada5c6)

- 논문의 **end-to-end architecture에 self-supervised pre-trained speech model을 포함**시키는 것은 **data limitation문제를 해결**할 뿐만 아니라, **어려운 case(eg. fully closed lips, /b/, /m/, /p/)들에 대한 mouth movements 정확도를 현저하게 향상**시킬 수 있다.

#### Extensive experiments and the user study to assess the quality of synthesized face motions

synthesized face motion의 성능을 평가하기 위한 광범위한 실험과 user study

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6b309f56-3598-4fe7-a752-a391de903c1d)

- 결과들은 2개의 3D datasets에 대해 **realistic facial animation과 lip-sync**의 관점에서 FaceFormer의 우수성을 입증한다.

# 2. Related Work

## 2.1 Speech-Driven 3D Facial Animation

 

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/84b68e8c-6d80-4a81-835a-f3d9af2727d8)

- Facial animation은 지난 몇년간 많은 관심은 받은 분야이다.
- 광범위한 2D-based approach들을 알고 있었지만, 논문은 해당 연구에서 3D model을 animating하는 것에 주력했다.
- 일반적으로, procedural(절차적인) method들은 talking mouth을 animating하기 위한 명확한 규칙들을 수립한다.
Ex) “dominance function”은 speech control parameters를 characterize하기 위해 사용된다. “dynamic viseme model(proposed by Taylor)”은 lip motion을 위해 phonemes의 one-to-many(일 대 다) mapping을 사용한다. “Xu”는 coarticulation(동시 조음) 효과를 modeling하기 위한 표준법을 설계하였다. “JALI(procedural approach SoTA model)”은 3D facial rig를 animate하기 위해 2개의 anatomical(해부학적인) action을 사용한다.

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bef06165-47c2-42f9-b5ef-5eb5c0e827ca)

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b36144a7-62e0-4dbc-b925-c5761b37916e)

- procedural method들의 매력적인 강점은 mouth movements의 정확도를 보장하는 system의explicit control이다.
⇒하지만 parameter tuning에 많은 수동적인 노력이 필요하다.(hyper-parameter)
- 위 방법 대신, “data-driven approach”의 다양한 방법들은 3D facial animation을 위해 만들어졌다.
- “Cao”는 Anime Graph sturcture와 search-based technique을 기반으로 3D facial animation을 synthesize한다.
- “sliding window approach”는 input으로 transcribed phoneme sequences가 필요하고, output을 다른 animation rigs로 나타낼 수 있다.
- “end-to-end convolutional network(Karras에 의해 구성된)”은 audio를 encoding하기 위해서 linear predictive coding method를 사용하고, facial expression의 다양성에 대한 차이를 분명하게 구별하기 위해 latent code를 design하였다.
- “Zhou”는 viseme(사람이 말하는 동안 얼굴과 입의 위치) animation curves를 예측하기 위해 phoneme groups, landmarks, audio features를 결합한 three-stage network를 사용한다.
- “VOCA”는 speaking styles의 다양성을 capture하는 speaker-independent 3D facial animation method이지만, generated face motion은 대부분 lower face(하관)에만 나타난다.
- 최근, “MeshTalk”는 categorical latent space를 학습한다.
⇒ audio-correlated face motion과 audio-uncorrelated face motion을 성공적으로 구분

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1db95c79-7423-4e04-a1d2-cf6c81d7662e)

- 논문의 연구와 관련된 method들은 high-resolution 3D data를 training에 사용하고, output이 3D vertex space의 high-dimensional vector로 표현된다.
- 위에서 언급한 model 중 2개는 short audio windows를 사용하여 training하기 때문에 long-term audio context를 고려하지 않고, 나머지 1개는 매우 사실적인 facial animation을 얻지만, 해당 모델은 animation quality와 unseen identity에 대한 generalization을 보장하기 위해서 매우 많은 양의 high-fidelity 3D facial data가 필요하다.

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0ecd4665-ad0c-4456-ab1e-ae536813cb46)

- “Transformer”는 RNN과 CNN의 강력한 대안으로 떠올랐다.
- **sequence tokens를 재귀적으로 처리하는 RNN**과 반대로, **transformer는 input sequence의 모든 token들을 병렬적으로 처리할 수 있어서 “long-range contextual information”을 효과적으로 modeling**할 수 있다.
- Vision Transformer(ViT)는 transformer를 image classification에 직접적으로 적용한 첫 연구이다.
- ViT를 따르는 다음 연구들은 image recognition문제들을 해결하기 위해 성능을 향상시키는 방법을 제안한다.
- 게다가, transformer-based models와 그 응용 방법들은 object detection, semantic segmentation, image generation 등의 분야에도 제안되었다.
- Computer graphics분야에서, transformer는 3D point cloud representation과 3D mesh를 위해 사용되었다.(ex. “Point Transformer”, “Point Cloud Transformer”, “Mesh Transformer”)

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6f6f6c2b-583e-4de9-b3d9-cb68d4d64e57)

- 3D body motion synthesis를 위한 가장 최근 연구들 중 몇몇은 sequential data modeling에서 transformer의 성능을 연구하고, 인상적인 결과를 만들어냈다.
- output motion이 상당히 제한적이지 않은(자유로운) dance generation와 다르게, **speech-driven 3D facial animation task는 본질적으로 audio와 face motion사이의 alignment가 필요**하다.
⇒ **정확한 lip motion을 구현**하기 위해서
- 한편, **long-term audio context는 주요하게 고려**될 것으로 예상된다.
⇒ **whole face를 animating하기 위해 중요**
- 결론적으로, **논문은 “speech-driven 3D facial animation” 문제를 해결하기 위해 적절한 성질들을 통합한 “FaceFormer”를 소개**한다.

# 3. Method(FaceFormer)

---

***Figure 2***

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/35349eb6-144d-43db-984f-1c3741f596bf)

- Figure 2는 “FaceFormer”의 전체적인 architecture를 보여준다.
- **Transformer를 갖는 encoder-decoder architecture**는 **“raw audio”를 input**으로 받고, **autoregressive하게 animated 3D face meshes sequence를 생성**한다.
- Layer normalization과 residual connection은 간소화를 위해 생략했다.
- **“FaceFormer encoder”의 전체적인 design은 “wav2vec 2.0”을 따른다.**
- 게다가, **audio features resampling을 위해 TCN뒤에 “linear interpolation layer”가 추가**되었다.
- 논문은 **“corresponding pre-trained wav2vec 2.0 weights”로 encoder를 initialize**하였다.
- **“FaceFormer decoder”는 2개의 main module로 구성**된다.
**1) biased causal MH self-attention with a periodic positional encoding**
⇒ longer input sequence에 대한 generalization을 위해
**2) biased cross-modal multi-head(MH) attention**
⇒ audio-motion modality의 alignment를 위해
- Training시에, **TCN의 parameter들은 고정되고, 다른 부분들은 모두 learnable**하다.

---

![Untitled 21](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c300565c-68d3-4417-8ee9-f771d2edf082)

![Untitled 22](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0a95d086-af64-4f6a-9485-2aa8f5f4c4fc)

- 논문은 **“speech-driven 3D facial animation”을 “sequence-to-sequence(seq2seq2) learning problem으로 공식화**하고, **새로운 seq2seq architecture를 제안**한다.(Fig2)
⇒ **“audio context”와 “past facial movement sequence”의 영향을 받는 “facial movements”를 autoregressive하게 예측**
- **“ground-truth 3D face movements($Y_T=(y_1,...,y_T)$)**”의 sequence($**T$는 visual frames의 수**)와 **corresponding raw audio($\chi$)**가 있다고 가정하자.
- 목표는 **raw audio($\chi$)가 주어졌을 때, $Y_T$와 유사한 facial movements($\hat{Y}_T$)을 synthesize할 수 있는 model을 만드는 것**이다.
- “encoder-decoder framework”에서, **encoder는 먼저 “raw audio($\chi$)”를 “speech representation($A_{T'}=(a_1, ...,a_{T'})$)”로 변환**한다. ($T'$은 speech representation의 frame 길이)
- **“style embedding layer”는 “speaker identities($S=(s_1,...,s_N)$)”를 나타내는 learnable embedding을 포함**하고 있다.
- 그 후, **decoder은 “speech representation($A_{T'}$), speaker($n$)의 style embedding($s_n$), past facial movements”에 conditioned된 facial movements($\hat{Y}_T$=($\hat{y}_1,...,\hat{y}_T$))를 autoregressive하게 예측**한다.

![Untitled 23](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f1f1bf6a-61d5-4b81-9bed-33f307a8ac84)

- 공식으로 나타내면 위 식(1)과 같다.
- $\hat{y}_t=FaceFormer_{\theta}(\hat{y}<t,s_n,\chi)$    —— (1)

$\theta$ : model parameter
$t$ : sequence내에서 current time-step
$\hat{y}_t\in\hat{Y}_T$

## 3.1. FaceFormer Encoder

### 3.1.1 Self-Supervised Pre-Trained Speech Model

![Untitled 24](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f846bd13-b291-44ac-aad6-c3a0e0236f95)

- **“FaceFormer encoder”의 구성**은 **self-supervised pre-trained speech SoTA model인 wav2vec 2.0을 따른다.**
- 구체적으로, **encoder는 “audio feature extractor”와 “multi-layer transformer encoder”로 구성**된다.
- **몇 개의 TCN(Temporal Convolutions Layers)로 구성된 “audio-feature extractor”**는 **‘raw waveform input’을 ‘frequency($f_a$)의 feature vectors’로 변환**한다.
- **“transformer encoder”는 ‘multi-head self-attention’과 ‘feed-forward layers’의 층으로 구성**된다.
⇒ **‘audio feature vectors’를 ‘contextualized speech representations’로 변환**
- temporal convolutions의 output은 quantiaztion module에 의해 유한한 speech units으로 discretized(이산화)된다.
- masked language modeling과 유사하게, “wav2vec 2.0”은 contrastive(대조하는) task를 해결함으로써 true quantized speech unit을 식별하기 위해 masked time step 주변의 context를 사용한다.

---

#### Temporal Convolutional Network(논문 외 추가적인 자료조사)

![Untitled 25](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ade666e8-6112-4963-af8d-1ca2b757aa42)

- causal convolution으로 구성된 architecture이다.
- TCN은 input으로 다양한 길이의 sequence를 받고, input과 동일한 길이의 sequence를 output으로 내보낸다.
- ***“Sequence Modeling이란?***
input sequence($x_0,...,x_T$)를 받아, 각 시간 t에 대응하는 output sequence($y_0,...,y_T$)를 예측하는 것. $y_t$를 예측할 떄, t시점을 포함한 이전의 input($x_0,...,x_t$)만을 활용한다.
⇒ $f: X^{T+1}\to Y^{T+1}$로 sequence modeling network를 표현할 수 있음
- $L(y_0,...,y_T,f(x_0,...,x_T))$를 최소화하는 function $f$를 찾는 것이 궁극적인 목표인 것
- ***“Causal Convolution”***

![Untitled 26](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/3ca609a4-a769-488b-afcc-cb6b27460450)

⇒ 미래의 정보 $x_{t+1}$이 $y_t$예측에 사용되지 않은다.(미래의 정보가 과거로 유출되지 않음→ Causal의 의미)

---

![Untitled 27](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c6e3afc1-1bbc-4b20-bfe7-5288891aa8e9)

- **FaceFormer의 encoder는 “pre-trained wav2vec 2.0 weights”로 initialize**되고, 그 다음에 **random하게 initialize된 linear projection layer로 구성**된다.
- facial motion data가 **“audio-feature extractor에 의한 frequency($f_a$)”**와 다른 **frequency($f_m$)으로 caputure**되기 때문에(”BIWI dataset기준으로 $f_a=49Hz, f_m=25fps)$, **audio features를 resampling하기 위한 “linear interpolation layer”를 temporal convolution 다음에 추가**한다.
⇒ **linear interpolation layer의 output length는** $kT(k= {f_a\over f_m})$
- 따라서, **linear projection layer의 output은 $A_{kT}=(a_1,...,a_{kT})$로 나타낼 수 있다**.
- 이렇게 하여, **audio-motion modality가 “biased cross-modal multi-head attention”을 통해 alignment**될 수 있다.

## 3.2. FaceFormer Decoder

---

### Positional Encoding이란(+ Transformer 동작 원리)

“출처: [https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)”

[트랜스포머(Transformer) 파헤치기—1. Positional Encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)

***(위 글만큼 Transformer에 대해 쉽게 정리된 글을 본 기억이 없다. 위 글의 내용과 자료를 바탕으로 Positional Encoding에 대해서 공부해보았다. 해당 정리의 출처는 모두 위 blog의 자료들이다.)***

![Untitled 28](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b121477b-70d3-44b5-845d-87a46c6593f1)

#### Input Embedding이란

- Input data를 컴퓨터가 이해할 수 있는 vector값으로 변환하는 과정이다.

![Untitled 29](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/25098b6a-2f74-469f-a0c8-f6e97a9d3323)

- 예시는 위와 같다. input data(sentence)가 주어지면, 각 단어들은 해당하는 index값에 mapping되어, 이 index값들이 Input Embedding에 전달되는 것

![Untitled 30](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/786d7ad8-2459-4ffb-9e08-9c3053d35277)

- 각 단어 index들은 서로 다른 vector값을 갖는다. (위 예시에서는 embedding의 크기가 4이지만, 실제 논문에서는 512)
⇒ 각 vector dimension은 각 단어의 feature값을 갖고, 단어들의 feature값이 유사할 수록, vector space에서의 embedding vector는 서로 가까워진다.

![Untitled 31](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/eb896b63-af34-4006-a2a5-f23d5b1a6ba1)

- 문맥상 유사도가 높아지면 embedding vetor의 값이 가까워지는 것이다.

#### Positional Encoding

- 우선 Postitional Encoding이 필요한 이유는

![Untitled 32](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8bca680a-c530-49aa-a696-524798336355)

- 위와 같이 LSTM이나 RNN의 경우, Input sequence가 model에 입력되면, 순서대로 data가 처리된다. (계산속도가 느리다)
⇒ Positional encoding이 없어도 자연스럽게 단어의 정보가 입력되는 것

![Untitled 33](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/55e710c3-38ec-4297-8b35-b5c6c6d7da49)

- 반면, Transformer는 병렬적으로 처리한다. 따라서 단어의 순서를 알 수 없다.⇒ Positional encoding이 필요

***“단어의 위치 정보가 중요한 이유”***

![Untitled 34](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/dd1d7d33-1f7f-491b-963d-0ca76ed1a122)

- 정말 직관적으로 이해하기 쉬운 예시를 blog에서 제시해주었다. 위 문장을 보면 not위치에 의해 1, 2번 문장의 의미가 완전히 뒤바뀐 것을 볼 수 있다. 따라서 단어의 위치 정보를 유지하는 것은 매우 중요하다.

![Untitled 35](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/979868ad-0053-44c6-9c12-299d2fa3e4d7)

- 위 그림과 같이 Positional Encoding을 통해 산출한 위치 정보를 input embedding에 더해주어야 한다.

***“Positional Encoding을 적용할 때 주의해야 할 점”***

반드시 지켜져야 하는 중요한 2가지 규칙이 있다고 한다.

1. 모든 위치값은 ‘sequence의 길이’ 또는 ‘input’에 상관없이 동일한 값을 가져야 한다.

![Untitled 36](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7d7d4071-c9b1-45db-8758-86a3cdc2c4e6)

1. 위치 값이 너무 크면, 단어 간의 상관관계와 의미를 유추할 수 있는 context정보가 상대적으로 작아지기 때문에, 너무 큰 값을 가지면 안된다.

![Untitled 37](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/5a5b0d6e-e96d-4a43-a5a6-0a3d2ae10cf1)

***“위치 벡터를 얻는 2가지 방법과 문제점”***

1)

![Untitled 38](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ce50ba74-9e66-4658-869b-c82756ea5667)

- 위와 같이 Linear하게 값을 위치 벡터를 지정하면, sequence 길이가 커질 수록 위치벡터의 크기가 커지고, 위치벡터가 특정한 범위를 가지고 있지 않아서 generalization이 불가능

2)

![Untitled 39](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/43550aba-5127-4d54-83c4-de7dd4bd9b6e)

- 첫 token에 0, 마지막 token에 1을 위치 벡터로 지정하고, 사이 값을 (1/단어수)으로 normalization하는 방법은 sequence의 길이에 따라 위치 벡터값이 바뀌고, token 간의 차이도 달라진다.

⇒ 위치 벡터 값은 너무 큰 값이 나오면 안되고, 같은 위치의 token은 같은 위치 벡터 값을 가져야 한다.

→ 이를 모두 해결하는 방법이 sine&cosine함수를 positional encoding에 사용하는 방법

#### Positional Encoding을 위한 Sine&Cosine 함수

- Sine&Cosine함수가 Positional Encoding의 조건에 만족하는가?

1) sine&cosine함수는 -1~1사이를 반복하는 주기함수이기 때문에 지나치게 큰 값을 갖지 않는다.

2) sigmoid함수를 사용하지 않는 이유는, sine&cosine함수는 주기함수이기 때문에 longer sequence가 들어오더라도 위치벡터 값의 차이가 작아지지 않지만, sigmoid는 sequence가 길어지면 위치벡터 간의 차이가 작아질 수 있다.

![Untitled 40](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1e20b056-2933-4588-9cff-1095e7b82d63)

3) 같은 위치의 token은 항상 같은 위치 벡터값을 가져야 한다는 조건에서 문제가 생길 수 있는데,

![Untitled 41](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7ede0dde-0a14-4ccf-aa05-e7565932cbaf)

위 예시 그림처럼 주기함수이기 때문에 position이 다른데, 같은 위치 벡터를 가질 수도 있는 문제가 발생한다. 하지만, 위치 ‘벡터’라는 것에 주목해보자.

![Untitled 42](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/070a3bea-30bc-4cb4-a1d2-a56bcc94a468)

위 예시처럼 각 벡터 차원마다 주기가 다른 함수를 사용하면, position이 다르면 겹칠 수 없다. 위 주기에 대한 공식은 아래와 같다.(position(i)값이 커질수록 frequency가 작아진다)

![Untitled 43](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d9d52533-2849-4cd6-a376-9a41c4bbdde4)

---

### 3.2.1 Periodic Positional Encoding

![Untitled 44](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ac653edd-f913-489b-a085-cd584b390793)

- 실제로, **transformer는 longer sequences에 대해 매우 한정된 generalization 성능**을 보인다.
⇒ **“sinusoidal positional encoding method”로 인해**
- **“ALiBi(Attention with Linear Biases)” method는 query-key attention score에 constant bias를 추가하여 generalization 성능을 향상**시키고자 하였다.
- 논문의 experiments에서, 직접적으로 **sinusoidal position encoding 대신 ALiBi를 사용하는 것은 inference과정에서 정적인 facial expression을 만들 수도 있다는 것을 발견**하였다.
- **ALiBi는 input representation에 “position information”을 추가하지 않는다.**
⇒ 이는 **temporal order(시간의 순서) information의 robustness에 영향**을 줄 수도 있다. 특히 **training sequence가 인접한 frame 사이에 적은 motion variation**을 가지고 있는 경우
- 위 문제를 완화하기 위해, 논문은 **“Periodic Positional Encoding(PPE)”**를 고안하였다.
⇒ **ALiBi를 유지하면서 temporal order information을 더할 수 있다.**
- 구체적으로, 논문은 원래의 sinusoidal positional encoding method를 **‘hyper-parameter $p$’(주기를 가르킨다)에 대해 periodic(주기적인)하도록 수정**하였다. 아래 식(2)와 같다.

![Untitled 45](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9cb2299b-2ee1-4edc-b436-f01cb406b728)

- $**t$: token position 또는 current time-step in the sequence**
- $**d$: model dimentsion**
- $**i$: dimension index**

![Untitled 46](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/09a3e677-7762-4d32-893b-f3920494fff4)

- 각 token에 대해 각각의 position identifer를 지정하는 것 대신, **“Periodic Positional Encoding(PPE)”는 주기적으로 각 period($p$)마다 position information을 더한다**.
- **PPE 전에, “motion encoder”를 통해 face motion($\hat{y}_t$)를 $d$-dimensional space로 project**한다.
- **speaking style에 따라 modeling**하기 위해, **“style embedding layer”를 통해 one-hot(0 또는 1로만 encoding된 data) speaker identity를 $d$-dimensional vector $s_n$으로 embedding**하고, 이를 **facial motion representation에 더한다.**

![Untitled 47](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/294e249d-cbc8-4f16-89d9-e535806a2f28)

- $W^f$는 weight, $b^f$는 bias, $\hat{y}_{t-1}$는 last time step의 prediction
- PPE는 temporal order information을 제공하기 위해 $f_t$에 적용된다.

---

### Multi-Head Attention에 대해 알아보자

이전에 Transformer 논문을 읽은 적이 있다. 하지만 인공지능을 공부한지 1달도 채 안되었을 때 읽었기 때문에, 그 당시에 이해했던 부분은 전체 논문의 10%도 안되었을 것이라고 생각한다. 또한 FaceFormer는 Transformer와 많은 연관이 있기 때문에, 다시 한번 복습한다는 생각으로 이번 기회에Transformer의 동작 원리에 대해서 복습하고자 한다.

감사하게도 Transformer의 원리, 특히 Multi-Head Attention에 대해 자세히 정리된 블로그 글이 있어서 해당 자료를 기반으로 이해를 해보았다.

해당 파트에서 정리된 자료들은 모두 해당 블로그의 출처를 가지고 있다.

**“출처: [https://www.blossominkyung.com/deeplearning/transformer-mha](https://www.blossominkyung.com/deeplearning/transformer-mha)”**

[트랜스포머(Transformer) 파헤치기—2. Multi-Head Attention](https://www.blossominkyung.com/deeplearning/transformer-mha)

- Attention이란 무엇일까? 우선 Attention의 개념이 LSTM에서부터 나왔다는 것은 처음 알게 되었다. 정말 간단하게 말하자면, 단어간의 유사도를 측정하는 것이다. Attention에 대한 설명은 아래 블로그를 통해 이해할 수 있었다.

[3. Attention [초등학생도 이해하는 자연어처리]](https://codingopera.tistory.com/41)

#### Self-Attention과 Attention

- 말 그대로 Self-Attention은 같은 문장 내에서, 단어들 간의 관계를 파악하기 위해 사용하는 방법이다. 동일 문장에서 token들 간의 유사도를 계산한다.

![Untitled 48](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/14ff7d12-b5be-4d06-9552-49ebdd95e04a)

- 해당 블로그에서 들었던 예시로는, ‘tear’이라는 동일한 단어가 있지만, 전자의 ‘tear’는 ‘paper’와의 유사도를 통해 ‘찢다’라는 뜻을 갖고 있고, 후자의 ‘tear’는 ‘shed’와의 유사도를 통해 ‘눈물’이라는 뜻을 가지고 있다는 것을 알 수 있다. 따라서 문장 내에서 self-attention을 통해 문장의 context를 더 잘 이해할 수 있게 될 것이다.

#### Query, Key, Value

- Transformer 논문을 처음 읽었을 때, 어려웠던 개념이 바로 query, key, value이다. 이번 기회에 확실하게 이해해보고자 한다.

“Query”: 물어보는 단어, Query가 Key의 단어들과 얼마나 연관이 있는지 알아보려고 하는 것이 목적

“Key”: 정보를 제공하는 단어들, Query와 얼마나 연관이 있는지 알려주는 역할

“Value”: 특정 key에 해당하는 input sequence의 정보로 가중치를 구하는 것에 사용

⇒ “Query”와 “Key”의 유사도에 따라 value값을 가져올 수 있는 것

#### Self-Attention

##### 1. Linear Layer

![Untitled 49](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bb123bdf-c07a-4878-a9b1-cdb25b68d5ff)

- embedding matrix+Positional Matrix가 Linear layer에 들어간다.
⇒ query, key, value 각각의 차원을 줄여 병렬 연산에 적합한 구조를 만듦

![Untitled 50](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/69b8280a-358c-40c0-a05d-37a9f151d5fd)

##### 2. Attention Score

![Untitled 51](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2356503e-b57f-4da1-917e-1753c3e2c822)

- Linear layer를 통해 얻은 query, key, value값을 통해 attention score를 얻는다.
- 위와 같이 Query와 Key의 내적을 통해 Attention Score를 얻을 수 있다. 자세한 연산은 아래 그림과 같다.

![Untitled 52](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fd0af19d-c7bf-49c7-9328-74af0a8b8eb4)

- 위 예시에서는 Query, Key의 내적값들이 Attention score matrix로 표현되는 것을 볼 수 있고, 각각의 값은 유사도와 같다.
- 식은 아래와 같이 나타낼 수 있다.

![Untitled 53](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8058fb26-54a8-4053-bd3a-f9b180763a5d)

- 위 식에 대한 의미를 자세히 살펴보면 아래와 같다.

![Untitled 54](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b33fdf0f-e79a-42d6-9f04-51ae8b0a5cc9)

- 위 식은 “cosine similarity”식이다. 두 벡터 A, B가 유사할수록 1에 가깝고, 다를수록 -1에 가까운 것을 알 수 있다. cosine similarity는 아래와 같은 식으로 표현할 수 있다.

![Untitled 55](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/54a4c046-f6d7-4860-be31-bb3ff365eeb7)

- cosine similarity는 벡터의 곱을 scaling으로 나눈 값이다. 따라서 최종적인 유사도 계산식은 아래와 같이 나타낼 수 있다.

![Untitled 56](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/3346061c-2902-47f9-a2c1-2043c6bd9b06)

- 유사도를 구하는 식을 위와 같이 간단하게 나타낼 수 있는데, 이를 Query와 Key로 나타내면 아래와 같은 식으로 나타낼 수 있다.

![Untitled 57](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6121060d-7839-4439-8f2a-7604558a1ef6)

- 위처럼 나타내면, Attention Score와 유사한 형태가 나오는 것을 볼 수 있다. 즉, 유사도를 softmax에 넣어 Value값에 곱해주는 것이다.

![Untitled 58](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6c830ca9-e92a-455c-a621-ad55a51c148e)

- 위처럼 Self-Attention에서는 자기 자신과 맵핑되는 값이 가장 크고, 그 다음으로 유사한 값이 크다는 것을 알 수 있다.

##### Scaling & Softmax

![Untitled 59](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bdd37d4e-0df0-42d0-bbc4-9edc7ab132cc)

- 이제 scaling과 softmax함수에 대해서 알아보자.
- Scale연산을 할 때, $d_k$는 sequence의 길이와 같다.
- Scaling이 필요한 이유는 dot product(내적)와 관련이 있다.
- dot-product의 특성상, sequence의 길이가 길어질수록 각각의 값이 커진다. 이렇게 되면, attention score를 softmax함수에 적용할 때, 특정값만이 과하게 남아버리고, 작은 값들은 gradient가 매우 작아져서 학습 성능이 나오지 않는다.
⇒ softmax가 비슷한 값들 사이에서 적용될 수 있도록 scaling을 해주면, softmax를 거쳐도 gradient가 유지된다.
- 이후, attention score를 0-1사이의 값으로 normalize하기 위해 softmax함수를 사용한다.

![Untitled 60](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/947ff0ff-77a7-44df-b4d2-42f25ab3a227)

- 마지막으로 Attention score와 value를 내적하면, 최종적인 attention value를 구할 수 있다.

#### Multi-head Attention

![Untitled 61](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c35f4792-bfdd-4369-a9fb-99ec7d4baaa7)

- Multi-head attention은 말그대로 여러번(h번)의 병렬 학습을 하는 것이다. 이렇게 동시에 학습시키는 이유는, 다양한 정보를 학습하기 위함이다. 예시는 아래와 같다.

![Untitled 62](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/64070d6b-d4f0-49cf-ab4a-91a0f43ff6bd)

---

### 3.2.2 Biased Causal Multi-Head Self-Attention

![Untitled 63](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/23438050-110f-4e58-a7f1-8b857333dca6)

![Untitled 64](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/84d8c0e7-d487-4f6f-b277-23dc5c62dbb5)

- 논문은 **ALiBi를 기반으로 “biased causal multi-head(MH) self-attention” mechanism을 고안**하였다.
⇒ **ALiBi는 language modeling에서 loner sequences에 대한 generalization에 좋은 성능**을 보인다.
- 주기적으로 encoding되는 **“facial motion representation sequence($\hat{F}_t=(\hat{f}_1,...,\hat{f}_t)$)”이 주어졌을 때, “biased causal MH self-attention”은 먼저 linear하게 $\hat{F}_t$를 “$d_k$차원의 queries ($Q^{\hat{F}}$)와 keys ($K^{\hat{F}}$)”와 “$d_v$차원의 values $V^{\hat{F}}$”으로 project**한다.
- **past facial motion sequence의 context에서, 각 frame 사이의 dependency를 학습하기 위해서, “scaled dot-product attention”을 통해 “weighted contextual representation”이 계산**된다.

![Untitled 65](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d40a2227-bcd3-4885-9004-c061c77f2952)

- $**B^{\hat{F}}$는 ‘temporal bias**이다.
⇒ **causality(인과 관계)를 확실히 하고, longer sequence에 대한 generalization 성능을 향상**시키기 위해 사용

![Untitled 66](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fd02dd7e-349f-4475-979b-63fe13b4b893)

- 구체적으로, **“temporal bias($B^{\hat{F}}$)”는 upper triangle부분이 매우 큰 음수인 matrix**이다.
⇒**current prediction에 future frames를 사용하지 않기 위해**
- **generalization 성능을 위해서, 논문은 $B^{\hat{F}}$의 lower triangle에 constant하고 non-learned한 biases를 추가**하였다.
- “ALiBi”와 다르게, **‘period $p$’를 도입하고, 각 period($[1:p],[p+1:2p],...$)에 ‘temporal bias $B^{\hat{F}}$’를 더하였다**.
- $**i$와 $j$를 $B^{\hat{F}}$의 index라고 정의할 때($1 \le i \le t$, $1 \le j \le t$), “temporal bias $B^{\hat{F}}$”는 위 식(6)과 같이 표현**할 수 있다.

![Untitled 67](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d536c5cd-fd85-4e47-b4e7-f4641b182cd3)

- 위와 같이, 논문은 **더 가까운 period에 더 높은 ‘attention weights’을 부여하여, “casual attention”을 편향**되도록 하였다.
- 직감적으로, **‘가장 가까운 facial frames preiod$(\hat{y}_{t-p},...,\hat{y}_{t-1})$’는 ‘$\hat{y}_t$의 current prediction’에 가장 많은 영향**을 미친다.
⇒ 따라서  논문에서 제안한 **‘temporal bias’는 ALiBi의 generalization된 형태**라고 볼 수 있고, $p=1$일 때 ALiBi는 특별한 case가 된다.

![Untitled 68](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/11099ba1-9c87-43a4-a63e-db7f6333ce7c)

- **‘MH attention mechanism’(H개의 parallel scaled dot-product attentions로 구성된)**은 여러개의 representation subspaces를 통해 구성된 complementary(상호보완적인) 정보들을 추출하기 위해 사용된다.
- **H heads의 output들은 concatenate**되고 **parameter matrix($W^{\hat{F}}$)에 의해 project**된다. 위 식(7)에서 볼 수 있다.

![Untitled 69](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f93231e8-61c5-4db6-9249-c3f9bc333352)

- ALiBi와 유사하게, 논문은 **“head-specific scalar($m$)을 MH setting을 위해 추가**하였다.
- **각 head$_h$의 temporal bias는 다음과 같이 정의된다.($B^{\hat{F}}_h=B^{\hat{F}}\cdot m$)**
- **scalar $m$은 “head-specific slope”이고, training시에 학습되지 않는다.**
- $H$**개의 heads가 있으면, $m$은 $2^{-2^{-log_2H+3}}$에서 시작하여, 각 element에 똑같은 $m$값을 곱하여 다음 element**를 계산한다.
- **구체적으로, model이 4 heads를 갖고 있다면, slopes는 $2^{-2},2^{-4},2^{-6},2^{-8}$**을 갖는다.

---

***“Figure 2”***

![Untitled 70](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8127d713-7068-4bc7-8610-5cad48c96db3)

---

### 3.2.3 Biased Cross-Modal Multi-Head Attention

![Untitled 71](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4246ca92-aff8-490d-8784-05379775d190)

- **“biased cross-modal multi-head attention”은 ‘Faceformer encoder의 outputs(speech features)’와 ‘biased causal MH self-attention(motion features)’를 결합**하는 것에 초점을 둔다.
⇒ **audio-motion modality를 alignment하기 위해**(Fig 2에 있음)
- 위 방법을 적용하기 위해, 논문은 **‘query-key attention score’에 “alignment bias”를 추가**하였다.
⇒간단하고 효과적임
- “alignment bias($B^A$)”는 위 식 (8)과 같이 나타낼 수 있다.

![Untitled 72](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8735076e-18ee-4282-9d1f-46d2f1daefca)

- $**A_{kT}$의 각 token들은 self-attention mechanism를 통해 long-term audio context를 capture**할 수 있다.
- 반면에**, ‘biased causal MH self-attention’의 output을 ‘$\tilde{F}_t=(\tilde{f}_1,...,\tilde{f}_t)$라고 할 때, $\tilde{F}_t$의 각 token들은 face motions의 history context를 encoding**한다.
- $**A_{kT}$와 $\tilde{F}_t$ 둘 다 “biased cross-modal MH attention”에 input**으로 들어간다.
- 이전의 과정과 마찬가지로, $**A_{kT}$는 2개의 matrix로 분리**된다.
⇒ ‘keys $K^A$’와 ‘values $V^A$’
- $\tilde{F}_t$는 ‘queries $Q^{\tilde{F}}$’로 변환된다.
- $V^A$의 weighted sum은 위 식(9)와 같이 계산된다.

![Untitled 73](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ff9d1963-029c-4908-ad60-e5e899a6aafd)

- 각각 다른 subspaces를 탐구하기 위해, 논문은 식(9)를 식(7)과 같이 H개의 heads가 있는 식으로 확장하였다.
- 최종적으로, **‘predicted face motion($\hat{y}_t$)는 “motion decoder”를 통해 $d$-dimensional hidden state에서 $V$-dimensional 3D vertex space로 project**되어 얻어진다.

## 3.3. Training and Testing

![Untitled 74](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ad059403-7cba-4ef1-bbd1-927c831f58a4)

- Training을 진행하는 동안, 논문은 “teacher-forcing scheme” 대신 “autoregressive scheme”을 사용한다.
- 논문의 실험을 통해, fully guided scheme(teacher-forcing)보다 less guided scheme(autoregressive)이 FaceFormer를 학습시킬 때, 더 잘 동작하는 것을 발견
- 우선 완전한 3D faccial motion sequence가 만들어지고 나면, FaceFormer는 “decoder outputs($\hat{Y}_t=(\hat{y}_1,...,\hat{y}_T)$)”와 “ground truth($Y_t=(y_1,...,y_T)$)” 사이의 MSE(Mean Squared Error)를 최소화하는 방향으로 training된다. (식(10)과 같다)
- $V$는 3D face의 vertex 개수이다.

---

#### Teacher-Forcing Scheme & Autoregressive Scheme

출처: [https://blog.naver.com/sooftware/221790750668](https://blog.naver.com/sooftware/221790750668)

##### 1. teacher-forcing scheme

![Untitled 75](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/56547b1c-7b71-413e-bb7a-7230e2d28a2f)

- 위처럼 seq2seq모델은 t-1의 예측이 t의 입력으로 들어간다. t-1의 예측값이 올바르지 않으면 잘못된 값이 도출될 수 있다. 이를 해결하기 위해 나온 방법이 teacher-forcing 이다.

![Untitled 76](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fe7ab0d0-475b-442f-a662-7338fe90908b)

- teacher-forcing은 위 그림처럼 ground truth를 입력으로 넣어주는 기법이다.

##### 2. Autoregressive scheme

출처: [https://otexts.com/fppkr/AR.html](https://otexts.com/fppkr/AR.html)

- autoregressive model은 변수 과거값의 linear한 조합을 통해 원하는 값을 예측하는 방법이다. 다음과 같은 식으로 나타낼 수 있다.

![Untitled 77](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/91230585-b0bc-4db5-b65c-c24662181613)

---

![Untitled 78](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d6a2c51c-778d-4ce8-9fd9-2a20631d43f7)

- Inference 단계에서, FaceFormer는 autoregressive하게 animated 3D face meshes sequence를 예측한다.
- 더 자세히 말하자면, 각 time-step마다, FaceFormer는 ‘raw audio($\chi$)’, ‘face motions history($\hat{y}_{<t}$)’, ‘style representations($s_n$)’에 따른 “face motion($\hat{y}_t$)”를 예측한다.(식(1))

![Untitled 79](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9090859c-efbf-4305-a7a9-f686bd547312)

- $s_n$은 ‘speaker identity’에 의해 결정되기 때문에, ‘one-hot identity vector’를 바꾸면 다른 style로 output을 바꿀 수 있다.