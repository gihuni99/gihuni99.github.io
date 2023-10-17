---
title: Transformer(Attention is All you Need)
date: 2022-09-25 20:00:00 +09:00
categories: [Paper, Transformer]
tags:
  [
    NLP,
    Paper
  ]
pin: true
---

# 사전 지식

## 1) 기존 Seq2Seq모델들의 한계점

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9b5e9ca1-7c13-4f29-9d57-4f3e884e8d7b)

위 사진처럼 모든 문장의 정보를 context vector “v”에 담고 있기 때문에 병목현상이 발생한다.

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/955ecca4-7bae-4319-8508-9edbee52af46)

디코더가 context vector를 매번 참고할 수 있도록 만들었지만 “v”는 여전히 고정되어 병목현상이 발생한다.

## 2) Seq2Seq with Attention

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1b54f81c-6afb-4082-b279-812e497f9613)

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a56a204d-1c3f-4768-9b4c-bd73d6c83ecd)

Decoder에서 출력될 때마다, Encoder의 출력 전부를 입력으로 받는다.

(=출력시 모든 문장을 고려한다)

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/356d00be-d480-4bae-825b-9ff5991102d1)

Decoder는 매번 출력할 때마다 Encoder의 모든 출력 중 어떤 정보가 중요한지 계산한다.

위 식을 보면 변수 i, j가 나오는데 내용은 다음과 같다.

 i : Decoder는 매번 한번에 하나의 단어를 출력하는데, 이때 출력하고 있는 단어

j : Encoder의 각각의 출력값

위 식에서 Energy에 대한 식이 나오는데, Energy는 문장의 모든 단어 h중에서 어떤 단어와 연관성이 있는지 수치적으로 알 수 있게 해준다. 따라서 이전의 Decoder가 출력했던 단어를 만들기 위해 사용했던 hidden state(s)와 Encoder의 각각의 hidden state를 고려하여 Energy값을 구한다. 이 Energy값을 통해 Weight를 계산한다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0ee10ed6-55ef-43be-b4e1-653954ed1f28)

# Transformer

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/60d64ad7-1a6a-4a99-a2d0-a32d4fae5694)

온전히 Attention mechanism에 기반한 구조이다.(Recurrence나 Convolution 사용x)

⇒ 더 parrallelizable하고, 훨씬 적은 학습 시간이 걸린다.

## -RNN의 단점

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/3d1fda8e-85c0-496a-8e00-ba52b5090bbd)

위는 RNN(Recurrent Neural Network)의 단점이다. 연속적인 계산(Sequential computation)문제로 input과 output간의 거리가 멀다. 따라서 병렬 계산(Parallelization)이 어렵다.

**⇒RNN과 CNN을 사용하는 대신 Positional Encoding 사용**

(Positional Encoding을 사용하여 별도의 위치 정보를 제공한다)

## Transformer의 동작 원리

### 1) Embedding

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/519dc315-7094-4955-a349-c36ce7a24b75)

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/178811b1-c79b-4925-bf1a-d884e3c51bc0)

문장을 embedding한다. 차원은 (단어의 개수)X(embedding차원) 이다.

 

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/42ffd550-e739-4d69-9938-01b373026b95)

RNN을 사용하지 않기 때문에 따로 위치정보를 제공해야 한다.

Embedding과 같은 차원의 Positional Encoding을 추가해준다.

### 2)Encoder

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/276a31cd-465a-485e-8eed-516059114caf)

Embedding후 Attention을 진행한다. Encoder에서의 Attention은 Self-Attention이다.

Self-Attention은 각각의 단어가 서로에게 어떤 연관성을 가지고 있는지 구하기 위해 사용한다. 다시 말해, 각 단어가 서로에 대한 Attention score를 구하여, 각각의 단어가 다른 어떤 단어와 얼마 만큼의 연관성을 가지는지 학습한다. 즉 문맥을 학습하는 것이다.

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f9077f43-2857-442c-94c8-aea1d95a6ec3)

Self-Attention이후 성능 향상을 위해 Residual Learning을 사용한다. 캡처화면에서 “Add”는 Attention 수행값과 Residual Connection을 이용한 원래의 값을 더한 것을 의미한다.

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d78acdeb-7368-4791-bf1a-ef10c1aef777)

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/5db95ba8-b02d-45a8-a009-c4e789b77ce2)

Feedforward Layer를 통과시키고, 지금까지의 같은 연산은 N번 반복한다.

### 3) Encoder & Decoder

![2471F9D9-914C-4096-B5AA-9B6C96E611D4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4181b5b6-698b-4217-8f91-7daac2d78938)

Decoder도 Encoder와 마찬가지로 여러 개의 layer를 갖는다. 따라서 각각의 Decoder는 Encoder의 출력값을 입력으로 받는다. 처음의 Multi-head Attention은 Decoder의 Self-Attention이고 두번째 Multi-head Attention은 Encoder-Decoder Attention이다. Self-Attention은 앞서 말한 Encoder의 Self-Attention과 동작이 같다. 

Encoder-Decoder Attention을 보면 가장 마지막에 나온 Encoder의 출력 값이 입력되는 것을 알 수 있다. 따라서 Decoder가 단어를 출력할 때마다 입력 Source중에서 어떤 단어에 초점을 두어야 하는지 전달받는 것이다. 즉, Encoder의 정보에 Attention하는 것이다. 예를 들어 “I am a teacher.”이라는 문장을 “나는 선생님이다.”라는 문장으로 번역해야할 때, ‘선생님’이라는 단어는 어떤 단어와 연관성이 높은지를 계산하는 것이다. 

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b7702e7a-1cef-4fa7-abdd-18b91dc3035b)

위 캡처화면을 볼 수 있듯이 Encoder의 가장 마지막 output이 각각의 Decoder의 input이 된다.

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/905378c7-5913-419c-a65d-5058c1c9f35e)

RNN에서는 입력 단어의 개수만큼 Encoder Layer를 거쳐서 hidden state를 만들었다. 반면 Transformer에서는 입력 단어가 모두 연결되어 위치 정보와 함께 한꺼번에 입력되고, Encoder를 한 번 거칠 때마다 병렬적으로 출력값을 구할 수 있다. 또한 Encoder의 출력값을 하나의 vector로 압축하는 과정도 없다. 따라서 계산 복잡도가 낮고, RNN이 없어도 된다.

### 4) Attention

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a82cc924-842a-4404-ab90-68af4dd40081)

Attention은 3가지의 input값들이 있다. Q(query), K(key), 그리고 V(value)이다. 

***Q(query) : 어떤 것을 물어보는 주체***

***K(key) : 어떤 것을 물어보는 대상***

***V(value) : 실제 값***

예를 들어 “I am a teacher”라는 문장에서 각각의 단어 ‘I’, ‘am’, ‘a’, ‘teacher’는 Key이다.

만약 ‘I’라는 단어가 각각의 단어와 얼마나 연관성이 있는지 물어볼 때, ‘I’는 Query이다.

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/3cfa057d-0dd3-41da-b29e-87af9b4fe6e4)

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/5ee99611-52ed-4bbf-8af0-92a3bfa3a0a0)

위 왼쪽 자료는 Scaled Dot-Product Attention으로 Q와 K를 이용하여 가중치, 즉 확률값을 구하고 V값과 연산을 통해 가중치가 적용된 Attention Value값을 출력한다.

Transformer에서 사용하는 것은 Multi-Head Attention인데, Scaled Dot-Product가 h개 있는 형태이다. 어떤 문장이 input값 V, K, Q로 구분되었을 때, 이 input값들이 h개로 구분된다. 따라서 Multi-Head로 h개의 서로 다른 Attention컨셉을 학습할 수 있다. 즉 다양한 특징들을 학습할 수 있는 것이다. 이후 Concatenate과정을 통해 다시 연결하여준다. input과 output의 demension이 같아야 하기 때문이다.

이처럼 각각의 Multi-Head Attention의 동작 방식은 같습니다. 하지만 각 위치마다 Q, V, K의 사용방식이 달라질 수 있다. 예를 들어, Encoder-Decoder Attention에서는 Decoder의 단어가 Q(query), Encoder의 출력 값이 K(key)와 V(value)가 된다.

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/088a7b1f-3893-41e3-97e8-26c07bcd7934)

위 캡처 화면은 이전에 설명했던 Multi-Head Attention의 과정을 식으로 나타낸 것이다. 

가장 위의 식은 Attention Value를 구하는 식이다.

그 아래의 식은 head의 개수, 즉 h개 만큼 Q, K, V를 나누어 Attention해주는 과정이다.

그 이후 Concat해준다.

### 4) Multi-Head Attention 동작원리

![Untitled 21](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/fc236a59-01a2-4aa5-989e-11a037937059)

![Untitled 22](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/62ad15df-7e80-498f-aba4-302e049a83df)

### 4-1) 실제 동작 과정

1) 하나의 전체 문장에 대해 Query, Key, Value로 Embedding한다. 아래의 자료를 보면 Q, K, V가 각 단어별 표현되는 것을 볼 수 있다.

![Untitled 23](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/84889b96-a547-4f95-a9e1-aefe559477de)

2) Attention연산을 통해 Scaled Dot-Product Attention을 실행한다. Q, K의 연산으로 Attention Energy를 구할 수 있다. 아래 그림처럼 각 단어별로 서로와 얼마나 연관성이 있는지 직관적으로 볼 수 있다. 이후 Softmax함수를 통해 확률적으로 가중치를 정하고 Value값에 곱하여 Attention연산을 해준다.

이때 mask matrix를 이용하여 특정 단어는 무시하도록 softmax함수가 0%에 가까워지도록 한다.

![Untitled 24](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1e877b7e-9149-47fd-a518-fd1e5f4db071)

![Untitled 25](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/27f3bccf-98af-44d0-b54f-b08e9c7e410a)

3) 아래 자료와 같이 각 head의 Attention 결과를 Concat하여 input과 output의 차원을 같게 한다. 

![Untitled 26](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/46d2d05d-5981-4836-baf0-07b096b1e33e)

![Untitled 27](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c5caf8a0-facd-4253-8bda-026f7fa9d0b9)

### 5) Attention의 종류

![Untitled 28](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b46e443a-033c-4185-803a-ee750ab7fa52)

Encoder Self-Attention은 각각의 단어가 서로와의 연관성을 구한다.

Masked Decoder Self-Attention은 단어를 출력할 때, 앞에 있는 단어만을 참고할 수 있게 한다. 예를 들어, “나는 축구를 했다.”라는 문장에서 ‘축구를’이라는 단어를 출력할 때, ‘했다’라는 단어를 참고하지 않도록 한다. 뒤의 단어를 참고하여 출력하는 것은 정답을 미리 알고 출력하는 것과 같기 때문.

Encoder-Decoder Attention은 Query가 decoder값이고, Key와 Value가 encoder값이다. Key와 Value가 Encoder값이므로 Encoder의 정보를 이용하여 Decoder의 출력값이 정해진다는 뜻이다.

### 6) Self-Attention

![Untitled 29](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b28f8c35-4f69-4281-be35-3ad8e49f6ffc)

## Transformer 마무리

![Untitled 30](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a314d8a1-65b7-4805-8bee-2d72f6e22263)

**출처:**

**1) 논문 ‘Attention is All You Need’(NIPS 2017)** 

**2) Youtube 동빈나 “[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)”영상**

**3) 인하대학교 ‘인공지능응용시스템-홍성은 교수님’ 강의 자료**
