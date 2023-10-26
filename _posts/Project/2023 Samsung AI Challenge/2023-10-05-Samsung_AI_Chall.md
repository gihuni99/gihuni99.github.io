---
title: 2023 Samsung AI Challenge:Camera-Invariant Domain Adaptation
date: 2023-10-05 00:00:00 +09:00
categories: [Project, 2023 Samsung AI Challenge]
tags:
  [
    Computer Vision,
    Domain Adaptation,
    Semantic Segmentation,
    Challenge
  ]
pin: true
---

## Challenge 배경
자율주행은 다양한 센서들을 사용해 주변 상황을 인식하고 이를 바탕으로 차량을 제어하게 됩니다. 카메라 센서의 경우, 장착 위치, 센서의 종류, 주행 환경 등에 따라 영상간의 격차(Domain Gap)가 발생합니다. 그간 여러 선행 연구에서는 이미지의 광도와 질감(Photometry and Texture) 격차에 의한 인식 성능 저하를 극복하기 위해, Unsupervised Domain Adaptation 기술을 광범위하게 적용해왔습니다. 하지만 대부분의 기존 연구들은 카메라의 광학적 특성, 특히 이미지의 왜곡 특성(Geometric Distortion)에 따른 영상간의 격차는 고려하지 않고 있습니다. 따라서 본 대회에서는 왜곡이 존재하지 않는 이미지(Source Domain)와 레이블을 활용하여, 왜곡된 이미지(Target Domain)에 대해서도 고성능의 이미지 분할(Semantic Segmentation)을 수행하는 AI 알고리즘 개발을 제안합니다.

## Challenge 주제
카메라 특성 변화에 강인한 Domain Adaptive Semantic Segmentation 알고리즘 개발

## Challenge 기간: 23/8/21~10/2

2023 Samsung AI Challenge에 참여하였다. 왜곡이 없고 labeling이 되어있는 dataset(source image)으로 학습을 시켜, 왜곡이 존재하는 target image에 대한 Semantic Segmentation성능을 향상시키는 것이다. Dataset의 구성은 다음과 같다.

### train_source_gt, val_source_gt
12개의 class로 labeling되어있고, 우리는 background의 class ID를 12로 설정하여 총 13개의 class에 대해 학습시켰다.
![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/d805fa5e-4330-4d4d-b7bb-018cba2be54f)

### train_source_image, val_source_image
왜곡이 존재하지 않는 training, validation image의 예시이다.
2048 x 1024 size의 image로 training 2193개, validation 466개로 구성되어 있다.
![VALID_SOURCE_211](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/b33a57aa-5b8c-4f9d-857f-92ff91ce33ca)

### train_target_image
왜곡이 존재하는 training image로 test image와 같이 fisheye camera로 취득되어 domain이 같다. 하지만 labeling이 되어있지 않다.
![TRAIN_TARGET_0743](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/42312f71-acde-4178-8c73-9630ee028695)

### test_image
fisheye camera로 취득되어 왜곡이 존재하는 test image이다.
![TEST_0284](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/e8acaca8-f761-4287-9b96-7aa70790a95a)

# 대회 참여 과정

## 아이디어
Domain Adaptation task를 처음 수행해봤기 때문에, 우선 어떻게 하면 주어진 상황에서 최선의 결과를 낼 수 있을지 고민해보았다. 

먼저, 주어진 data를 최대한 효과적으로 이용할 수 있는 방법에 대해 고민해보았다. Geometric Distortion이 없는 training dataset에는 labeling이 존재하고, supervised learning이 가능했다. 하지만 고민했던 부분은 **train_target_image**를 어떻게 활용하느냐였다. test와 같은 domain을 가지고 있지만 labeling이 되어 있지 않아 supervised learning에 사용할 수 없었다. 따라서 우리 팀은 **semi-supervised learing**을 사용하자는 결론을 내렸다.

그 다음 고민한 것은, 어떻게 domain adaptation을 효과적으로 수행할지였다. 결론지은 것은 Augmentation을 활용하는 것이었다. Geometric Augmentation을 효과적으로 사용하면, 왜곡이 없는 dataset으로 학습을 시키더라도 왜곡이 있는 test data에 대해 sematic segmentation을 성능 높게 수행할 수 있을 것이라고 생각했다. 

위에서 나열한 생각들을 모두 부합한 model이 **'AugSeg'** model이었다. AugSeg는 semi-supervised learning model중에서 좋은 성능을 가지고 있었다. FixMatch방법으로 teacher model과 student model을 각각 train_source_image와 train_target_image로 학습시키고, unlabeled loss와 labeled loss를 모두 학습에 관여시켜 최종적으로 student model이 학습되도록 한다. 여기서 중요한 점은 data pre-processing과정에서 Augmentation을 적용한다는 점이다. target data에는 강도가 약한 weak augmentation을, source data에는 다소 강도가 강한 strong augmentation을 적용한다.

## AugSeg / Weights&Biases tool

![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/20d06afb-0357-441e-9f23-b9fef8d82160)

위는 AugSeg의 Architecture를 표현한 것이다. AugSeg는 Backbone network와 Decoder를 선택할 수 있는데, 처음 성능측정을 진행할 때에는 Backbone으로 **resnet101**, Decoder로 **DeepLabV3+** 를 사용하였다. 우선 pre-trained model은 사용하지 않았고, Augmentation과 Batch size, learning rate등의 parameter값들도 변경하지 않았다. 기본적인 모델의 성능을 평가하기 위함이다.

Test는 따로 test dataset에 대한 ground truth가 존재하지 않는다. 다만 RLE 인코딩을 통해 모델이 생성한 segmentation map을 .csv파일 형태로 만든 후 dacon홈페이지에 제출하면 정확도를 확인할 수 있었다. 처음 결과를 확인했을 때에는 생각보다 성능이 좋지 않았다.

![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/44ec3eb4-4de6-4ace-8dc9-306cab16befc)

평가 기준은 mIoU로 0.03정도가 나왔다. 50epoch으로 overfitting되었다고 생각을 해봐도, 평가가 어떻게 이루어지는지 확인해보려고 SegFormer를 학습시켜서 얻은 결과 mIoU 0.14보다도 결과가 안좋아서 당황했다. base모델을 바꿔볼까도 고민해봤지만, AugSeg의 방법론이 실효성이 있을 것이라고 생각했고, 조금 더 공부하며 알아보고 싶었다. 또한 Domain Adaptation을 위한 기법을 전혀 적용하지 않았기 때문에 어쩌면 당연한 결과라는 생각을 했다. AugSeg는 Domain Adaptation을 위한 model이 아니라, unlabeled data를 사용하기 위한 semi-supervised learning model이기 때문이다. 따라서 우리 팀은 AugSeg이 Domain Adaptation task에 적합하도록 수정하고, 적절한 hyper-parameter값 등을 찾아 좋은 성능을 도출해보기로 했다.

추가적으로 우리 팀은 정확도를 수치뿐만 아니라, 이미지상으로도 실시간으로 확인하고 싶어서 **Weights and Biases** tool을 사용하여 loss, learning rate 등의 수치변화 뿐만 아니라 모델이 예측하는 segmentation map을 직접 이미지 상으로 확인할 수 있었다. 지금까지 TensorBoard만 사용하다가 이번 프로젝트에서 팀원이 적용하여 처음 W&B를 써보았는데, 정말 편리했다. 다른 프로젝트도 계속 사용할 것 같다. W&B의 결과 중 하나는 아래 그림과 같다.
![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/04f8b644-3642-4857-bdbc-b5547d4f9a20)
![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/e6b14880-848f-4324-8b19-012af33ede5e)

## 성능개선1(Pre-trained Model)

우선 가장 쉽고 효과적으로 성능을 올릴 수 있는 방법은 pre-trained모델을 사용하는 것이라고 생각했다. 따라서 ImageNet으로 pre-train된 resnet101을 사용하였다. 그 결과는 아래와 같다.

![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/0a13aade-fb5e-42df-aa90-5395361f1bc0)

생각보다 성능이 잘나왔다. 하지만, 2epoch학습을 시키는 것보다 8epoch학습시킨 성능이 현저히 낮았다. 따라서 0.41이라는 결과는 Domain Adaptation이 잘된 것이 아니라, 순수하게 pre-trained model의 성능이라는 것을 알 수 있었다.

## 성능개선2(Geometric Augmentation & Hyper-parameter)

논문에서는 teacher model, 즉 unlabeld data로 학습하는 모델의 data pre-processing과정에서는 Augmentation을 거의 적용하지 않았다. 하지만 student model의 data pre-processing과정에서는 Augmentation을 다소 강하게 주었다. 우리는 이러한 기조는 유지하기로 했다. AugSeg에서 Augmentation을 적용하는 방법은 여러가지의 Augmentation중에서 k라는 factor에 따라 k가지의 random한 Augmentation을 적용하는 것이었다. 하지만, Augmentation의 종류 중 Rotate를 제외하면 Geometric Augmentation이 존재하지 않았다. 우리의 목표는 Domain Adaptation을 하는 것이므로 해당 부분을 수정해야 했다. **Albumentations**를 이용하여 perspective, affine 등의 Augmentation을 적용하였고, geometric하지는 않지만 Fourier Domain Adaptation을 수행하는 FDA 등 다양한 Augmentation기법들을 추가하였다. 논문에서는 factor k가 클수록, 즉 많은 Augmentation을 한꺼번에 적용할수록 성능이 좋다고 주장하였지만 우리는 논문과 마찬가지로 k=3으로 고정하였다.

추가적으로 k와 같이 우리가 정해주어야 하는 hyper-parameter들이 많았다. 우선 lr, batchsize, weight decay 등의 parameter는 논문과 유사하게 설정하였다. hyper-parameter를 미세하게 조정하며 성능을 개선하는 작업은 이후에 진행하였다. learning rate scheduler는 **Cosine Annealing Warmrestarts**를 사용하였고(성능이 잘나온다는 사례가 가장 많았음), Loss function은 Cross Entropy Loss를 사용하였다.

추가적으로 가장 신경썼던 parameter는 Loss factor이다. student model과 teacher model은 각각의 loss값을 가지는데, 최종 loss는 이 2개의 값을 합친다. 논문에서는 supervised loss와 unsupervised loss를 그냥 더했다. 하지만, 실험을 여러번 돌려보았을 때 teacher model이 target image에 대한 잘못된 예측을 하게 되면, unsupervised loss의 영향력이 커서 잘못된 방향으로 학습을 하게 된다는 것을 발견했다. 따라서 우리는 unsupervised loss에 대한 factor를 0.5로 설정하였다. 이는 0.1, 0.9 등 다양한 값으로 학습을 시켰을 때, 가장 성능이 좋았던 factor값이다. 결과는 아래와 같다.

![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/d37a83e3-a95a-4c94-a79e-cc1649d93d95)

## 성능개선3(Backbone&Decoder)

마지막으로 시도한 방법은 AugSeg의 Framework(student-teacher framework)는 유지하면서, Base model을 바꾼 것이다. 지금까지는 **ResNet101 + DeepLabV3+**의 조합을 model로 사용하였다. 물론 Segmentation 성능이 매우 훌륭하고 보장된 조합이다. 하지만 우리는 **hierarchical transformer encdoer**를 갖는 **SegFormer**를 base model로 사용한다면 더 나은 결과를 도출할 수도 있을것이라는 생각을 했다. 결과적으로 pre-trained SegFormer를 student, teacher model을 사용했을 때, 우리 팀이 가장 높은 정확도를 도출할 수 있었다.

![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/f0c35525-a64c-45f9-85d6-f53b41c79e40)

이 외에도 여러 learning rate, weighted decay, Augmentation기법 추가 등 더 나은 성능을 만들기 위해 시도했지만, mIoU 0.467이 우리 팀의 최대 성능이었다.

## 최종결과

대회를 진행하는 중 확인할 수 있었던 score는 public data에 대한 정확도였는데, 대회 기간이 마감되고 private data의 결과를 확인했을때 mIoU가 0.487로 조금 더 높았고, 212팀 중 30등을 했다.
![image](https://github.com/gihuni99/2023_SamsungAIChallenge/assets/90080065/bba37cff-f266-47f1-8dd1-d35988e1d57b)

## 대회를 마치며..

이번 대회에서 많은 경험을 쌓을 수 있었다. Semantic Segmentation task는 연구실 인턴을 하며 꽤 다루어보았지만, challenge한 문제를 해결하기 위해 Segmentation model을 수정해보았던 적이 없었기 때문에 어려웠지만 많이 배울 수 있었던 대회였다. 특히 Geometric distortion을 해결하기 위한 방법론을 찾아보며 공부하는 과정이 즐거웠다. 이번 문제를 해결하기 위해 가장 주요하게 생각했던 부분은 Model선정이었는데, label이 없는 target image를 쓰면서 domain adaptation을 수행할 수 있도록 변형할 수 있는 모델이 어떤 것들이 있을지 찾아보면서 다양한 semi-supervised segmentation model들을 알아갈 수 있었다. AugSeg는 그중에서도 가장 적합하다고 생각되었던 모델이고, semi-supervised learning은 이론적으로만 알고 있었지만 어떻게 training과정이 이루어지는지 직접 다루며 확인할 수 있어서 좋은 기회가 되었던 것 같다. 특히 FixedMatch라고도 불리는 **Student-Teacher Framework**를 처음 보았는데, 흥미롭게 느껴졌다. 추가적으로 Source data와 Target data간의 데이터적인 차이를 분석하고, Domain adaptation을 위해 적절한 Augmentation을 추가하면서 유의미한 성능 향상을 이끌어낸 것이 이번 대회에서 내가 가장 열정적으로 실험에 임하고 재밌게 했던 부분이었다.

비록 좋은 성능을 내지 못해 순위권안에 들지는 못했지만, 많은 것을 배워가는 Challenge였다. 아쉬운 부분은 Challenge는 결국 성능을 최대한으로 도출하는 것이 목표인데, 너무 하나의 모델 framework를 신뢰하고, 성능 하락의 원인을 hyper-parameter와 같은 작은 부분으로만 생각을 한 것 같다. semi-supervised learning방식도 target data를 활용하는 것은 좋지만, 잘못 사용한다면 오히려 target data를 활용하지 않는 것보다 더 좋지 않는 성능을 낼 수도 있다는 생각을 대회기간 중에서는 하지 못했다. 이러한 실패를 경험삼아 앞으로는 조금 더 넓고 다양한 시선에서 문제를 해결하는 태도를 가질 수 있을 것 같다. 또한 1달이 조금 넘는 기간동안 내가 할 수 있는 모든 공부와 실험을 다 해보았기 때문에 후회는 없는 것 같다. 결과와 무관하게 나를 많이 성장시켜준 프로젝트였다.

[Dacon 2023 Samsung AI Challenge](https://dacon.io/competitions/official/236132/overview/description)

[Project Github](https://github.com/gihuni99/2023_SamsungAIChallenge)

[AugSeg Github](https://github.com/ZhenZHAO/AugSeg)