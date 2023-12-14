---
title: 23-1 Capstone Design
date: 2023-06-20 00:00:00 +09:00
categories: [Project, Capstone Design]
use_math: true
tags:
  [
    Computer Vision,
    Diffusion Models,
    Crowd Counting
  ]
pin: true
---

# 주제: Latent Diffusion Models를 이용한 Crowd Counting (Crowd Counting Using Diffusion-Based Latent Space)

# 기간: 2023/2/27 - 2023/6/23

## 주제 선정 배경

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d8971dc3-56f7-4cbd-91b5-ef32eb683d4a)

당시 이태원 압사 사고로 인해 Crowd Counting 기술에 대한 관심도가 커졌다. 나도 연구실 인턴을 진행하며 다양한 Crowd Counting 모델들의 논문을 공부하였다. 특히 Multi-modal Crowd Counting 모델들을 많이 보았다. 모델들을 직접 train, test해보고, 데이터셋들을 찾아보며 공부했다. 

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/54c7c8ab-b6dc-4dc8-8001-e40ecfe9c95a)

하지만 생각보다 데이터셋이 많지 않았다. 다른 Multi-modal dataset들도 마찬가지지만 데이터가 생각보다 부족하다는 것을 인지할 수 있었다. 단일 RGB 이미지만을 사용한 Crowd Counting 연구를 진행한 논문들도 많았지만, 당연하게도 Multi-modal Model에 비해 정확도가 많이 떨어졌다. 따라서 Crowd Counting은 여전히 Challenge한 문제라는 것도 알 수 있었다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/a287b1fb-8db0-432d-b6ed-170905c54a71)

그래서 단일 RGB이미지만으로 높은 정확도의 Density Map을 생성하고 싶다는 생각을 하게 되었다. 단순하게 Single RGB Crowd Counting을 생각한 것이 아니라, 생성형 모델을 사용하여 이 문제를 해결해보고 싶었다. 당시 Diffusion Model들을 흥미롭게 보고 있었고, 사람의 눈으로 구별할 수 없는 완벽한 이미지를 만들어내는 놀라운 성능을 확인하였다. 그래서 '생성형 모델이 이미지를 완벽하게 만들어 낸다면, Depth나 Thermal 같은 데이터도 만들 수 있지 않을까?'라는 생각을 하게 되었다. 이러한 주제를 구현할 수 있겠다는 생각을 하게 된 것이 '[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)'를 보고 난 후 이다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/4a71fec5-f8f6-4f18-9550-c062aabee78c)

위와 같은 모델의 구조를 활용하면 Conditioning Input으로 주고 Ground Truth Density Map을 모델의 Input으로 주어 효과적으로 학습시킨다면, 단일 RGB이미지만으로 정확한 Density Map을 생성하고 Crowd Counting을 할 수 있을 것이라는 가정을 하였다. 이것이 주제를 잡게 된 첫 시작이었다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/16021e58-4024-44dc-ab28-50b047ab7ea8)

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/155a4906-e262-47d7-b76e-ef867de09679)

내가 생각한 모델의 학습 흐름을 간단하게 나타내자면 위 이미지와 같다. 하지만 문제는 선행 연구가 없었다는 점이다([현재는 존재한다](https://github.com/dylran/DiffuseDenoiseCount)). 졸업프로젝트였고, 결과가 보장되지 않았다. 사실 학부생 수준의 연구로는 결과를 기존의 연구보다 높은 정확도로 낼 수 없을 것이라고 생각했다. 하지만 정말 흥미로운 주제를 생각했고, 가설로 세운 주제가 실제로 가능할지 너무 궁금했다. 그래서 좋은 결과가 나오지 않더라도 해당 주제로 연구를 진행하기로 하였다.

## 연구과정

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/ac752717-c1db-4cba-9c05-46b4e1964447)

Training은 ShanghaiTech RGB-D 데이터셋을 사용하였고, 주어진 Depth Data를 통해 Ground Truth Density Map을 생성하여 사용하였다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/31bbfb46-4c71-4709-b43a-967606945484)

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d691f397-0c7c-4753-8fcf-2d8fb2021878)

위 image는 Latent Diffusion Model의 학습과정을 Tensorboard를 통해 확인한 이미지이다. image에 noise를 추가하고, 제거하는 과정을 반복하며 Ground Truth Density Map에 가까운 데이터를 만들 수 있도록 학습한다. 이 때 핵심은 Conditioning Data(RGB image)가 Denoising과정에 관여하면서, RGB와 Density Map의 상관관계를 학습할 수 있을 것이라고 생각했다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/dee411a7-c0eb-4054-b9e8-d2d1cce20ffb)

### Issue

#### 1. 복잡한 Latent Diffusion Models의 코드

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/170d7f75-273f-4dd7-8692-65b841040d12)

나는 모델의 성능을 확인하기 위해 Inference는 많이 해봤지만, 모델의 코드를 내가 원하는 task에 맞도록 수정해본적이 없었다. 따라서 모든 코드가 새로웠고, 특히 Latent Diffusion Model의 코드가 정말 복잡하고 어려웠다. 따라서 주석을 달며 공부를 했고 코드를 분석하고 어떻게 동작하는지 이해하는 것만으로 2주라는 시간이 소요되었다. 지금 생각해보면 생각보다 많은 도움이 되었던 것 같다.

#### 2. Dimension 충돌

어떻게 보면 위와 같은 맥락이었다. 코드가 복잡하여 Density Map과 RGB Image가 어디서 차원이 변경되고, 또 어떻게 변경되는지 찾는것은 어려운 문제였다. 특히 Conditioning Data(RGB)가 Encoder를 거치고, 학습에 적용될 때 Concatenate방식과 Cross-attention방식 두가지 중 선택할 수 있었는데, 각 Encoder마다 Output이 다르고 적용방식이 달랐기 때문에 각 방법을 사용할 때의 적절한 차원값을 찾아야 했다.

#### 3. Density Map의 데이터 특성

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/d1afc2d9-eb20-4df7-a9c0-e7840cacff6a)

Density Map은 전체 픽셀 값을 모두 합쳤을 때, 사람의 수가 나온다. 따라서 실제로 Ground Truth Density Map을 이미지로 출력하면 검은색 이미지가 나온다. 픽셀값이 거의 1을 넘지 않기 때문이다. 위 이미지는 Density Map의 픽셀값을 255배한 것이다. 하지만 학습을 돌리면서 Density Map을 그대로 학습시키면, 이미지를 생성하는 Diffusion Model의 특성상, 원본 Density Map의 픽셀값이 매우 작기 때문에 제대로 생성하지 못한다는 것을 알 수 있었다. 따라서 Density Map에 큰 수를 곱하여 위 이미지처럼 눈에 보이도록 조정하여 데이터로 사용하였다. 따라서 기존 방식처럼 Density Map의 픽셀값을 더하여 Crowd Counting을 하는 것이 아닌 위치기반의 Crowd Counting을 하는 것으로 방향을 수정하였다.

#### 4. Cross-Attention VS Concatenate

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/b031e9da-c145-453e-a784-74d40c7a4475)

위에서도 언급했듯이 Conditioning Image를 적용할 때, Concatenate와 Cross-attention방식을 선택할 수 있었다. Cross-attention 방식으로 학습했을때의 효과성을 알기 때문에 해당 방식을 채택하려고 했다. 또한 실험을 하기 전 가설을 세웠을 때에는, RGB이미지와 Density Map의 연관성을 더 효과적으로 학습할 수 있을 것이라고 생각했지만 실험결과는 달랐다. 3번 Issue에서 Density Map을 변형하였기 때문에, 사람의 위치를 유지하는 것이 매우 중요했다. 하지만 Cross-attention방식은 사람의 위치를 학습하지 못했다. 이 원인은 Cross-attention을 적용할 때 이미지를 token화하는데, 이 때 위치에 대한 정보를 잃어 학습하지 못하는 것이라고 판단했다. 따라서 Concatenate방식을 채택했다. 여기서 아쉬운점은 Cross-attention을 적용하면서 위치 정보를 유지할 수 있도록 구현했다면 더 좋은 결과를 도출할 수 있지 않았을까 라는 아쉬움이 든다.

#### 5. 데이터 처리과정에서의 문제

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/3166b59e-86ba-4daf-91da-219438b2f341)

위 이미지는 사람이 많은 데이터를 모델에 넣었을 때 Inference결과이다. 처음 결과를 확인했을 때, 당황했다. 생각보다 학습능력이 많이 떨어졌다. 따라서 어떤 것이 문제인지 분석했다. 학습과정에서 문제가 생겼다고 생각했고, 그것이 Conditioning파트에서의 문제인지 Diffusion 과정에서의 문제인지 다양한 가능성을 열어두고 원인을 분석했다. 결과적으로 원인은 다른 곳에 있었다. 원인은 정규화에 있었다. 처음 주제에 대해 연구를 진행할 때에 픽셀값이 너무 작아 학습이 잘 되지 않는 것으로 판단했고, 0-1의 범위를 갖는 픽셀값을 0-255의 범위로 늘려 학습시킨 후 다시 0-1까지 정규화시켰다. 하지만 위에 언급했듯이 픽셀값을 크게 키워 위치기반으로 Crowd Counting을 하기로 결정을 했는데, 해당 정규화 코드를 지우지 않은 것이다. 따라서 생성된 Density Map의 픽셀값이 0-1사이의 값으로 바뀌었고, 이를 Auto Encoder에 통과시켜서 제대로된 결과가 나오지 않은 것이었다. 이 문제가 데이터 처리 과정에서의 오류일 것이라고 생각하지 못했고 학습이 잘 되지 않는 것이라고 판단했다. 따라서 다양한 parameter들을 바꾸어가며 많은 실험을 했고, 그만큼 많은 시간을 썼다. 해당 문제를 해결하는 것에 시간을 많이 소비하여 아직도 아쉬움이 남는다.

## 결과

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/6e283398-72de-474c-a60d-d9f05ccafe07)

위 결과는 실제 학습시킨 모델로 Test한 결과이다. 언뜻보면 결과가 좋은 것 같지만, 정확한 사람의 위치를 찍지는 못하는 모습이다. 사람이 없는 곳은 Density Map의 point가 찍히지 않지만, 각 사람들의 머리위치를 정확하게 인식하고 Density Map을 생성하지는 못했다. RGB 이미지와 Density Map사이의 연관성을 어느정도 학습하고 결과를 도출하였지만, 좋은 성능을 내지는 못했다.

![image](https://github.com/gihuni99/Capstone-Design-2023-1-/assets/90080065/ec20ba54-24cd-48ba-b6b0-902ea19b28b7)

실제 성능표를 확인해보아도 위 결과처럼 우수한 성능을 보이지 못한다. 하지만 의미있다고 생각한 부분은 Diffusion Model을 활용하여 데이터를 생성할 수 있는 가능성을 보았다는 점이다. 

## 소감

처음으로 스스로 주도하여 진행하였던 연구였다. 공부했던 내용을 바탕으로 해결하고 싶었던 주제가 생겼고, 보장된 결과가 없었지만 도전하고 싶었다. 처음해보는 프로젝트인만큼 부족한 부분이 많았던 것 같다. 사소한 문제였음에도 며칠을 투자하며 원인을 분석하려고 했고, 생각보다 허무하게 해결했던 기억이 많다. 그만큼 낭비했던 시간들이 많았던 것 같다. 하지만 부족했던만큼 더 배우고 공부하려고 노력했고, 문제를 해결하는 과정에서 책으로는 배울 수 없었던 것들을 많이 배웠다. 원인은 전혀 다른 곳에 있었고 엉뚱한 부분을 해결하려고 계속 시도했을지라도, 새롭게 알고 배운 것이 많은 것 같다. 무엇보다도 연구에 대한 열정을 알게 된 것 같아 기뻤다. 얕은 학부생 수준의 프로젝트였을지라도, 발견하게 된 문제와 그것을 스스로 생각한 방법으로 해결하고 개선하기 위한 모든 과정이 연구인 것 같다. 쉽게 해결되지 않고 막히는 부분이 있을 때에는 괴로운 감정도 들었지만, 그것을 이겨내고 해결해나가는 과정이 의미있었다. 여러모로 많은 것을 배우게 된 프로젝트이다. 


[Capstone_Gihub주소](https://github.com/gihuni99/Capstone-Design-2023-1-)

[프로젝트_논문파일](https://drive.google.com/file/d/1x9UTBVQ9gdK-3d9HEtfeqr5-axvcbsZO/view?usp=share_link)

[주제발표자료](https://docs.google.com/presentation/d/1HPqhqRphsLtik-fHVErgOFiD9X_9s-xN99jXQazjWFk/edit?usp=share_link)

[중간발표자료](https://docs.google.com/presentation/d/1T_Xj3mcZ3Jxo2vgtsl6M5D6pUCiGPUmqeAr3dnC8eCY/edit?usp=share_link)

[최종발표Template](https://docs.google.com/presentation/d/12pnczMlaHSltxF1JA5VBSMAWnfiHctuo0RWz1FXIUHM/edit?usp=share_link)