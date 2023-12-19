---
title: LaughTalk:Expressive 3D Talking Head Generation with Laughter 논문 리뷰
date: 2023-12-19 00:00:00 +09:00
categories: [Paper, 3D Talking Head Generation]
use_math: true
tags:
  [
    Paper,
    3D Talking Head Generation
  ]
pin: true
---

# Abstract

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/58292f03-084f-44b3-80a3-b05f31050bbf)

- ‘Laughter’는 고유한 expression이다.(긍정적 interaction에 필수적임)
- 최근 ‘3D talking generation method’들은 그럴듯한 ‘verbal articulations’를 만들어내지만, ‘vitality’와 ‘laughter, smiles’의 세부사항들을 capture하는 것에 실패한다.
- 해당 논문에서는, 새로운 task를 제시한다.

⇒ ‘articulate speech’와 ‘authentic laughter’ 모두 가능한 3D talking head generation

- 논문에서 새롭게 엄선한 ‘dataset’은 ‘pseudo-annotated, human-validated 3D FLAME parameters and vertices’와 paired된 ‘2D laughing videos’로 구성되어 있다.
- 논문에서 제안된 dataset이 주어졌을 때, 2-stage training 전략을 사용한다.
    - model은 먼저 ‘talking’을 학습하고, 그 후에 ‘laughter’ expression 능력을 획득한다.

# 1. Introduction

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a1c264cc-63e3-4123-a676-02dfbce3d50e)

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fb3fe482-a43c-4d36-8b6e-49067d1e18e2)

- ‘speech-driven 3D facial animation’은 꾸준히 주목받고 있다.
- 또한 deep learning분야를 통해 최근 많은 발전을 이루었다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/36002ee3-de31-48de-9dd5-4eadfaa7d428)

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/dadbfeda-876b-46e2-922f-e6dd3a47bbcb)

- 이전 연구들은 ‘speech’의 ‘verbal signals’에 대해 정확한 ‘lip synchronization’을 얻는 것에 초점을 두었다.

⇒ ‘human interaction’뿐만 아니라 ‘human-computer interaction’에도 필수적인 ‘non-verbal signals’에 대해서는 소홀했다.

- ‘non-verbal’ 단서들 중, 특히 ‘laughter’는 다양한 이유들 때문에 중요하다.
- 따라서, ‘non-verbal expressions’를 포함하는 ‘speech-driven 3D facial animations’를 synthesize하는 것은 ‘human-robot interaction’에서 중요한 발전이 될 것이다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9b8fff1a-6d89-4ad6-b4bb-469086f43183)

- 그럼에도 불구하고, ‘speech’와 ‘laughter’이 모두 있는 ‘3D facial expression animation’은 매우 challenge하다.
- 먼저 ‘speech’와 ‘laughter’를 모두 포함하는 datasets의 부족이 첫번째 어려움이다.
    - 기존 ‘3D scan, speech paired dataset’은 통제된 실험 환경에서 취득되었다.
    
    ⇒ ‘real-world scenario’에는 필요한 다양성과 자연스러움이 부족
    
- 두번째로, ‘verbal’, ‘non-veral’ cue가 speech안에 함께 얽매어 있다.
    - 이는 명확한 ‘verbal’, ‘non-verbal’ cue를 extract하기 어렵게 하고, ‘3D talking head’가 그들을 동시에 expression하도록 training시키는 것이 challenge하다.
- 위 문제들을 해결하는 것이 ‘speech’와 ‘laughter’를 모두 포함하는 ‘3D facial animation’을 해결하는 가장 중요한 안건이다.

---

***Figure 1***

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/5efa094b-d343-4fac-a6d9-53d44a206b55)

- Learning to laugh and talk
- 논문은 ‘verbal, non-verbal expressions’를 포함하는 ‘speech-driven 3D talking head generation’을 task로 제시한다.
- ‘verbal signals’에 대한 정확한 ‘lip-sync’를 주요하게 초점을 두었던 기존 연구들과 다르게, 해당 논문의 목표는 ‘synchronized lip movements’와 ‘synchronized laughter’를 동시에 나타내는 것이다.

---

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/23bf45fc-5bff-4da1-b16c-589532495bb7)

- 논문에 연구에서는 새로운 task를 제시한다.
    - ‘talking’, ‘laughter’가 모두 있는 ‘speech-driven 3D face animation’
- 논문의 주요 초점은 ‘synchronized accurate lip movements’와 ‘synchronized laughter’를 동시에 나타낼 수 있도록 model을 training하는 것이다. (Fig 1)
- 위 task를 위해, 논문은 “LaughTalk dataset”을 collecting하였다.

⇒ ‘laughter’와 함께 다양하고 자연스러운 ‘speech’를 포함하는 ‘in-the-wild 2D videos’와 ‘corresponding 3DMM(FLAME) parameters’로 구성되어 있다.

- 2D video datasets의 풍부함에도 불구하고, ‘speech’와 ‘laughter’를 모두 포함하는 video를 ‘parsing’하는 것은 간단한 일이 아니다.
    - video parsing은 video를 분석하고 정보를 extract하는 과정
- 논문은 ‘laughing, talking videos parsing’이 가능한 ‘data curation pipeline’을 고안하였다.
    - curation은 정보를 수집, 정렬하여 제공하는 과정
    - ‘non-active speakers video’, ‘scene changes’, ‘abrupt head movements’ 등의 noisy data를 filtering 할 수 있다.
- 위 2D videos는 3D 정보를 포함하고 있지 않기 때문에, ‘3D face reconstruction models’, ‘exemplar fine-tuning method’를 사용한다.

⇒ 2D videos dataset에 연관된 ‘reliable, robust pseudo GT FLAME parameters’를 생성하기 위함

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f9236232-676b-451a-8d59-7628cdc8b5a0)

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/e900d477-8cea-4e0a-a363-e8d052d3ed93)

- 위 dataset이 주어졌을 때, 논문은 “LaughTalk”이라고 불리는 baseline model을 design했다.
- LaughTalk은 ‘2-stage training’절차를 사용한다.
    - 처음에는 ‘speech articulation’을 training
    - 그 다음에는 ‘laughter expression’을 training
- First stage
    - ‘pre-trained audio encoder’를 통해 ‘audio feature’를 extracting
    - ‘Transformer decoder’가 ‘audio feature’로부터 ‘FLAME’ parameters를 regression하는 것을 training
        - 위 training에서는 ‘LaughTalk dataset’의 일부인 ‘$LaughTalk_{MEAD}$(neutral(중립적인) speech와 facial movements)’를 사용
        
        ⇒ ‘verbal cues’에만 초점을 두었다.
        
- Second stage
    - ‘separate model’이 ‘residual FLAME parameters’를 regression하도록 training
    
    ⇒ ‘first stage model’은 training되지 않은 부분
    
    - second stage에서 사용된 ‘separate model’은 ‘first stage’에서 사용된 모델과 동일한 architecture를 갖지만, weights는 공유되지 않는다.
    - 이제, ‘LaughTalk dataset’의 나머지 일부인 ‘$LaughTalk_{CELEB}$(speech와 laughter를 모두 포함)’을 사용한다.
    - ‘first stage model’이 ‘speech articulation’을 나타내도록 학습되었을 때, ‘second stage model’은 ‘non-verbal signal’을 expression하는 것에 초점을 두고 training된다.
    - 각 ‘stage(1, 2) model’에서 파생된 ‘FLAME parameters’를 combining하는 것은 ‘3D facial animation’에서 ‘verbal’, ‘non-verbal’ cue가 동시에 나타날 수 있게 해준다.

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/01e16e01-db72-4477-89ae-f0850d42aa22)

- 논문은 기존 ‘3D talking head models’와 비교하며 ‘LaughTalk’의 능력을 입증한다.
- 공평한 비교를 위해, 기존 model들도 논문에서 제안하는 dataset으로 학습시켰다.

⇒ 기존 모델들도 talking과 laughing이 가능하도록

- 논문은 model이 ‘synchronized non-verbal cues’를 나타내는 지 평가하기 위해 ‘pre-trained emotion feature extractor’를 사용하였고, ‘speech articulation’을 평가하기 위해 ‘lip vertex error’를 측정하였다.
- 논문의 실험은 “LaughTalk”가 ‘synced laughter generation’에 뛰어나다는 것을 입증할 뿐만 아니라, 기존 method들과 비교하여 좋은 ‘lip articulation’을 갖는다는 것을 보인다.
- 논문의 주요 기여도
    - ‘speech articulation’과 ‘laughter’를 동시에 expression하는 ‘3D talking head generation’ task를 소개한다.
    - “LaughTalk dataset”을 collect, curate한다.
        - ‘speech’와 ‘laughter’가 있는 ‘2D videos’와 ‘corresponding pseudo GT FLAME parameters’로 구성되어 있다.
    - “LaughTalk” baseline과 ‘2-stage training’전략을 제안한다.
        - ‘speech’로부터 ‘verbal, non-verbal signals’가 있는 ‘expressivve 3D faces animation’을 training

# 3. Learning to laugh and talk

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bc45484f-5448-45e8-b6f7-0abd1bf12ff8)

- Sec. 3.1: 전체적인 내용 설명
- Sec. 3.2: ‘2D video’와 ‘3DMM parameters’로 구성된 “LaughTalk dataset’ 설명
- Sec. 3.3: ‘laughing’이 포함된 ‘speech-driven 3D talking head generation model’ “LaughTalk”제안

## 3.1. Overview

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4fbdcfe9-32d2-4c20-bd08-13d32ce4031b)

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1a0bb858-4adf-4c72-8aa7-ff40149b4605)

- 논문의 목표는 주어진 ‘speech audio’를 통해 ‘speech’와 ‘laughter’가 포함된 ‘3D face animations sequence’를 synthesize하는 것이다.
- ‘global emotion labels’로 명확한 conditioning을 했던 기존 방법과 다르게, 논문에서는 ‘speech’안의 ‘talking(verbal)’, ‘instantaneous laughing(non-verbal)’ cue에 synchronize된 ‘3D faces’의 ‘lip’과 ‘expression’을 나타낸다.
- 예를 들어, ‘speech’가 ‘neutral tone’으로 전달되면, ‘3D face’는 ‘neutral expression’을 유지하고 ‘lip animating’에만 초점을 둘 것이다.
- 반대로, ‘speech’안에 ‘laughing’이 있다면, ‘3D facial features’는 동시에 animate할 것이다.

⇒ ‘laughter’와 관련된 ‘expression’의 특징적인 ‘upward shift’가 묘사될 것이다.

- 새롭게 task를 개척하는 단계이기 때문에, 논문은 적절한 ‘3D face dataset’과 ‘baseline model’을 만드는 것에 중점을 두었다.

### Preliminary

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/73a0b1a4-0f33-4d15-bbda-d7ecbffb1bca)

- “FLAME(parametric 3D head model)”을 ‘LaughTalk dataset’과 ‘3D talking head’에 대한 ‘3D human face representation’으로 사용하였다.
- ‘face shape coefficients($\beta \in \R^{\|\beta\|}$)’, ‘facial expression coefficients($\psi \in \R^{\|\psi\|}$)’, ‘pose($\theta \in \R^{3k+3},k=4$ joints)’가 주어졌을 때,
- “FLAME model $M$”으로, ‘3D face mesh $V \in \R^{n_j \times 3}(n_v=5023)$’와 ‘3D facial landmarks $J^{3D} \in \R^{n_j \times 3}(n_j=68)$’를 획득할 수 있다.
    - $[V,J^{3D}]=M(\beta,\psi,\theta)$

## 3.2. LaughTalk dataset

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fec1fdbf-aac4-47f4-b4fa-91a588470db8)

- ‘talking, laughing facial video clip’과 ‘corresponding pseudo-annotated FLAME parameters’로 구성된 “LaughTalk dataset”을 제시한다.
- 먼저 ‘neutral talking’을 위해 “MEAD”에서 ‘2D video clips’를 curate한다.⇒ $LaughTalk_{MEAD}$
- ‘laughter’가 있는 ‘talking’을 위해 “CelebV-HQ”와 “CelebV-Text”를 curate한다.
⇒$LaughTalk_{CELEB}$
- 그 후, 유의미한 sample들을 얻기 위한 filtering을 하고, 마지막으로 ‘3D pseudo-annotation’을 취득한다.
- ‘curated dataset’의 예시는 Fig2에 있다.

---

***Figure 2***

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7af7e59b-8ce8-4887-91f3-9fee90b05b6c)

- LaughTalk Dataset

---

### Data collection

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8e4441f2-3a33-42ff-b26f-00aa51140b90)

- ‘CelebV-HQ’와 ‘CelebV-Text’는 ‘rich facial attributes(appearance, emotion, action)’가 있는 ‘in-the-wild face videos’를 가지고 있다.
- 논문에서는 “$LaughTalk_{CELEB}$”의 ‘laughing, talking video clips’를 construct하기 위해 “laugh”, “smile”, “happy”, “talk”를 찾는다.
- 유사하게, “MEAD”는 ‘emotion attributes’와 ‘emotion intensity’의 annotation을 제공한다.

⇒ “MEAD”에서는 ‘neutral attribute’를 찾고, “$LaughTalk_{MEAD}$”의 ‘neutral talking video clips’를 모은다.

### Data filtering process

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/dcb7df4a-f101-4b16-a0c1-0e6e1d02412d)

- data collection 후에, ‘valid, clean dataset’을 구축하기 위해 ‘noisy samples’를 filtering한다.
- 먼저, ‘video clips’가 항상 ‘laughter’를 포함하도록 하기 위해, “laugh detector”를 사용하여 ‘laughter’가 3.5초 이상 지속되지 않는 sample들을 filtering하였다.

⇒ original dataset에서 정확하지 않은 annotations이 되어 있는 sample들을 줄일 수 있다.

- 두번째로, “LaughTalk dataset”은 반드시 ‘talking faces’가 포함되어야 한다. 하지만, 몇몇의 video clip들은 ‘scene’밖의 speech를 포함하고 있다. 따라서 “active speaker detector”를 사용하여 ‘non-active speakers video’를 filtering하였다.

⇒ ‘speech’와 synchronize된 ‘facial video’를 얻기 위함

- 세번째로, ‘scene transitions(화면 전환)’의 video clips를 잘라내기 위해 “scene detector”를 사용하였다.

⇒ 화면이 전환되는 video가 포함되는 것을 방지할 수 있다.(3.5초 이상되는 videos에 대해서만 해당 process를 적용)

- 마지막으로, ‘individual’s face’가 정면으로 보이지 않고 일부분만 보이거나, ‘abrupt head movements’를 보이는 video sample들을 제외하였다.

⇒ 결과적으로 ‘clear, frontal facial shots’만을 남겨두었다.

### Lifting 2D video to 3D

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4d635d28-2ba0-494d-b045-56a359c470e1)

- cleaned ‘2D in-the-wild videos’를 획득한 이후에, ‘video clips’로부터 ‘audio’와 ‘facial movements’에 synchronize된 ‘3D faces’를 reconstruction하였다.
- 하지만, 기존 ‘3D face reconstruction models’는 ‘video’로부터 일시적으로 consistent하고 accurate한 ‘3D face meshes’를 reconstruction하는 것에 한계가 있다.
- ‘SOTA face reconstruction models’는 ‘static 2D images’에 대해서만 학습된다.

⇒ ‘rare(드문) poses’의 face를 추론하는 것에 한계가 있고, ‘per-frame independent inference(frame마다 독립적으로 inference됨)’ 때문에 ‘jittered motion’을 발생시킨다.

- 위 문제를 해결하기 위해, “EFT”에 영감을 받아, ‘3D face meshes’를 ‘neural network parameters’로 re-parameterize하는 ‘optimization method’를 사용하였다.
- neural network를 “SPECTRE”로 initialize하고, 각 video clip에 대해 optimization하였다.

⇒ accurate, robust ‘pseudo GT FLAME parameters’를 획득하기 위함

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d693ada4-e280-4e9a-be0b-1b46ccfa2f30)

- 모든 processing 이후, 943개의 ‘video clips’, ‘corresponding pseudo-annotated FLAME parameters’를 취득하였다.

---

***Table 1***

![Untitled 21](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fedcdebc-e7fe-4ee4-997f-761dc8ceaf4d)

- ‘training set video clips’에 대해서는 3.5초가 되도록 잘라냈지만, ‘test set’은 다양한 input에 대해 평가하기 위해, 다양한 길이가 되도록 남겨두었다.

⇒ test set의 평균 길이가 training set보다 길다

---

## 3.3. Two-stage training baseline: LaughTalk

![Untitled 22](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0823f007-4711-4319-8950-aab6c01b445f)

- ‘verbal signals’와 ‘non-verbal signals’는 ‘speech’안에 복잡하게 얽혀있기 때문에, ‘3D talking head model’이 ‘laughter’와 ‘speech’를 동시에 animate하도록 training하는 것은 challenge하다.
- 위를 해결하기 위해, task를 ‘sub-problems’로 나눔으로써 접근하였다.
- baseline model “LaughTalk”의 overview는 Fig3에 있다.

---

***Figure 3***

![Untitled 23](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/daf93607-7dfa-4c46-82da-d0f45a729ac0)

- ‘stage-1 model’은 ‘input speech’에서 ‘verbal features’를 extract하고, autoregressive 방식으로 ‘facial motion representations’를 생성한다.
- 동시에, ‘stage-2 model’은 위와 똑같은 ‘speech’에서 ‘non-verbal features’를 extract하고 ‘residual facial motion representations’를 생성한다.
- 위 두개의 ‘representations’는 ‘element-wise’로 더해지고, 그 후에 ‘3D face animations’를 synthesize하기 위해 “FLAME model”의 input으로 들어간다.

---

![Untitled 24](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4767a7b7-c264-4855-a3bf-6379c4728366)

- “LaughTalk”는 ‘2-stage training strategy’를 수행한다.
- 먼저, ‘stage-1 model’은 ‘talk(speech의 ‘verbal signals’)’를 training한다.

⇒ ‘neutral speech videos($LaughTalk_{MEAD}$)’를 통해 ‘lip movement synchronization’을 위한 ‘facial representation’을 생성하도록 training된다.

- ‘stage-1 model’이 ‘speech-related movements animation’ ability를 획득한 후에, ‘$LaughTalk_{CELEB}$’를 사용하여 ‘stage-2 model’을 training한다.

⇒ ‘lip movements’와 ‘facial expressions’를 동시에 animate할 수 있도록 training

- ‘stage-1 model’의 parameters를 freeze하였기 때문에, ‘stage-2 model’은 ‘stage-1 model’에 의해 학습되지 못한 ‘residual facial representations’를 생성할 수 있도록 training된다.
- 위 ‘residual aspects’는 ‘speech’안의 ‘non-verbal cue’와 관련되어 있을 것이다. (’cheek movements’, ‘facial expressions’와 같은)
- ‘residual representations’를 ‘pretrained stage-1 model’의 output과 결합함으로써, “LaughTalk”는 ‘synchronized verbal, non-verbal signals’를 동시에 포함하는 ‘3D talking head’를 animate할 수 있다.
- 이제 ‘task formulation’, ‘detail model architecture’, ‘각 stage의 training objectives’에 대해 간략하게 소개한다.

### Task formulation

![Untitled 25](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/39550737-9ce2-4bbc-a20b-502e32a7ef0c)

- $F_{1:T}=(f_1,...,f_T)$는 ‘facial motions의 temporal sequence’
    - $f_t$는 ‘facial representation’
        - ‘facial representation($f_t$)’는 “FLAEM parameters”의 ‘expression coefficient’와 ‘jaw pose’의 concat ⇒ $f_t=[\psi_t,\theta^{jaw}_t]$
- $A_{1:T}=(a_1,...,a_T)$는 ‘speech representations의 sequence’
- 논문의 목표는 corresponding ‘speech representation sequence($A_{1:T}$)’를 통해 ‘facial representations($F_{1:T}$)’를 순차적으로 synthesize하는 것이다.
- $F_{1:T}$를 ‘mesh vertex’로 visualize하기 위해, 임의적인 ‘face shape parameter($\beta$)’와 함께 “FLAME model($M$)”의 input으로 들어간다.
- animated vertex $V_{1:T}=M(\beta,F_{1:T})$

### Stage-1: learning to talk

![Untitled 26](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c6e6fb17-a536-4a6f-8e19-95acb2e11204)

- ‘stage-1 model’은 주로 ‘speech-related signals’를 capture하여 ‘3D faces animation’을 training하도록 설계되었다.
- ‘stage-1 model’은 다음과 같이 구성된다.
    - Verbal Encoder $E_v$
        - ‘input audio’로부터 ‘speech-related features’를 extract
    - Transformer Decoder $D_v$
        - ‘speech-related feature’를 통해 ‘autoregressive manner’로 ‘facial representations sequences’를 생성

![Untitled 27](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f6a9e170-881f-4db9-8b6e-bb0a7f742dbc)

![Untitled 28](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/85a5aba3-2584-4993-8109-3c4ebe44a9f5)

- “Faceformer”을 따라, 논문에서는 ‘Verbal Encoder’으로 ‘wav2vec 2.0’을 사용한다.

⇒ ‘wav2vec 2.0’은 ‘audio feature extractor’와 ‘multi-layer transformer encoder’를 포함

- ‘audio feature extractor’는 ‘temporal convolutional network(TCN)’을 활용

⇒ ‘speech’의 ‘raw waveform’을 ‘feature vector’로 변환

- 그 후, ‘Transformer Encoder’는 ‘audio features’를 ‘speech representations’로 변환한다.
- ‘Transformer Decoder’는
    - ‘causal self-attention’으로 구성되어있다.
    
    ⇒ ‘이전 facial representations’의 context안에서의 dependency를 training하기 위해
    
    - 그리고, ‘cross-modal attention’을 사용한다.
    
    ⇒ ‘audio’와 ‘facial representations’를 align하기 위해
    
- 위 과정을 공식으로 나타내면 아래와 같다.
    - $\hat{f}_t=D_v(E_v(A_{1:T})$,$\hat{F}_{1:t-1})$
        - $\hat{f}_t$는 ‘currently predicted facial representation’
        - $\hat{F}_{1:t-1}$은 ‘past predicted sequences’
- 모든 sequence $\hat{F}_{1:T}$를 predict한 후에, ‘mesh vertex’와 ‘3D landmarks’로 변환하기 위해 임의의 ‘shape parameter($\beta$)’와 함께 “FLAME model($M$)”의 input으로 넣는다.
    - $[\hat{V}_{1:T}$,$\hat{J}^{3D}_{1:T}]=M(\beta$,$\hat{F}_{1:T})$

![Untitled 29](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/44cb3312-e24b-4e81-b0db-27faaeb014b6)

- 논문은 ‘wav2vec 2.0’의 ‘pretrained weights’로 설정된 ‘frozen TCN’를 유지하며, ‘stage-1 model’의 ‘Transformer Encoder, Decoder’를 training한다.
- 주로 ‘verbal signals’에 초점을 둠으로써 model이 ‘facial movements’를 animate하도록 training시키기 위해서, ‘$LaughTalk_{MEAD}$’를 사용하여 training하였다.

⇒ $LaughTalk_{MEAD}$는 ‘neutral tone’의 speech를 포함하고 있다.

- ‘stage-1 model’의 ‘training objective’는 위 식(1)과 같다.
    - $\hat{\psi_{1:T}}$는 ‘predicted facial representations($\hat{F}_{1:T}$)’의 ‘expression parameters’
    - $\hat{J}^{3D}_{1:T}$는 ‘predicted 3D facial landmarks’
    - $\{\lambda_*\}$는 각 ‘loss term’의 weights

### Stage-2: learning to laugh

![Untitled 30](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/db14a7c1-dcd6-4b98-b455-d879b20cd9e3)

![Untitled 31](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4f8cd4bd-4683-4f21-8e02-33ec98b0f1f7)

- ‘stage-2 model’은 ‘pretrained stage-1 model’을 포함하고, ‘speech’와 ‘laughter’를 모두 포함하는 ‘3D faces animation’을 training하도록 설계되었다.
- ‘stage-1 model’을 freeze하기 때문에, ‘stage-2 model’은 ‘stage-1 model’이 학습하지 못했던 ‘residual facial representations generation’을 training하는 것에 초점을 두도록 유도된다.
- 논문에서는 위와 같은 ‘residual representations’가 ‘speech’의 ‘non-verbal cues’와 연관되어있을 것이라고 추정한다.
- 다시 말해, ‘stage-2 model’은 ‘residual facial representations($\hat{F}'_{1:T}$)’를 예측하도록 training되는 것이다.

⇒ ‘stage-1 model($\hat{F}_{1:T}$)’의 output과 결합되면, ‘verbal, non-verbal signals’가 모두 express되는 ‘3D face’를 생성할 수 있다.

![Untitled 32](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2efafc0f-c4f8-4c78-a129-ea7f533a8eb2)

- ‘stage-2 model’의 architecture는 ‘stage-1 model’과 동일하다. (다음과 같이 구성됨)
    - Non-verbal Encoder $E_n$
    - Transformer Decoder $D_n$
- ‘stage-1 model’과 다른 점은, ‘stage-2 model’은 ‘emotion recognition model’의 ‘pretrained weights’로 initialize된 ‘wav2vec 2.0’을 ‘Non-verbal Encoder’로 사용한다는 것이다.

![Untitled 33](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bde46aef-5708-4d33-868b-8c0c656edc79)

- ‘Transformer Encoder, Decoder’를 ‘laughter’와 ‘speech’가 모두 있는 ‘in-the-wild audio samples’를 포함하는 “$LaughTalk_{CELEB}$”를 사용하여 training한다.
(’TCN’과 ‘stage-1 model’은 frozen)
- ‘stage-2 model’의 training objective는 위 식(2)와 같다.
    - $V_{1:T}, \hat{V}'_{1:T}$는 각각 ‘GT vertex sequence’, ‘predicted vertex sequence’
- ‘predicted vertex($\hat{V}'_{1:T}$)’는  ‘pre-trained stage-1 model’과 ‘stage-2 model의 residual part’를 통해 나온 “FLAME parameters’를 합쳐서 만들어졌다.
- 공식으로 나타내면 위 식과 같이 나타낼 수 있다.

---

***Figure 4***

![Untitled 34](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/59cf20c1-9905-4739-baef-d81e0f25a068)

- 각 stage의 model들에 의해 synthesize된 ‘output mesh’를 visualize한 결과이다.
- ‘stage-1 model’은 ‘input speech’의 ‘lip movements’를 출력한다.
- ‘stage-2 model’은 ‘non-verbal signals’의 ‘expression을 보완하고, 더 정확한 ‘lip movements’가 가능하도록 한다.

---

![Untitled 35](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1ee5749f-ff30-40dc-8276-7071a35de79b)

![Untitled 36](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7d6567c6-7d15-48b4-ac1e-3c20749c5dc8)

- ‘stage-2 model’을 통해 training되는 ‘residual representations’의 영향력을 분석하기 위해, 각 model(stage-1,2 model)을 통해 생성된 ‘facial representation’을 ‘mesh vertex’로 변환하였다.
- Figure 4.는 ‘stage-1 model’, ‘residual model’, ‘final stage-2 model’의 output을 시각적으로 보여준다.
- ‘stage-1 model’의 output이 ‘speech-related facial motion’을 animating하도록 설계된 반면, ‘residual output’은 더 expressive한 ‘3D faces’를 만드는 것에 기여한다.
- 위 ‘residual representations’는 2가지 역할을 한다.
    - ‘non-verbal laughter signal’을 나타내는 것
    - 동시에 ‘lip articulations’의 성능을 올리는 것
- 위 발견들은 ‘2-stage training approach’의 효율성을 지지하고, ‘speech’와 ‘laughter’를 동시에 갖는 ‘3D talking head animation’을 성공적으로 이끈다.

### Training details

![Untitled 37](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/19e2e50f-88d1-4e36-95a3-fdb56d41036a)

# 6. Conclusion

![Untitled 38](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/60c47f38-3bda-4257-ac01-6d718e3b0114)