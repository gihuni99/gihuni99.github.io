---
title: 1208. [S/W 문제해결 기본] 1일차 - Flatten (SWEA)
date: 2024-11-13 00:00:00 +09:00
categories: [PS(SWEA), D3]
use_math: true
tags:
  [
    PS(SWEA),
    Python
  ]
pin: false
---

※ SW Expert 아카데미의 문제를 무단 복제하는 것을 금지합니다.

출처: https://swexpertacademy.com/main/code/problem/problemDetail.do?problemLevel=3&contestProbId=AV139KOaABgCFAYh&categoryId=AV139KOaABgCFAYh&categoryType=CODE&problemTitle=&orderBy=RECOMMEND_COUNT&selectCodeLang=PYTHON&select-1=3&pageSize=10&pageIndex=1

# 나의 풀이

```python
T = 10
for test_case in range(1, 11):
    D=int(input())
    b_list=list(map(int,input().strip().split()))
    for _ in range(D):
        max_b=max(b_list)
        min_b=min(b_list)
        if max_b-min_b==1 or max_b==min_b:
            break
        idx_max_b=b_list.index(max_b)
        idx_min_b=b_list.index(min_b)
        b_list[idx_max_b]-=1
        b_list[idx_min_b]+=1
    max_b=max(b_list)
    min_b=min(b_list)
    ans=max_b-min_b
    print(f'#{test_case} {ans}')
```

- 최대 길이가 100이기 때문에 max, min을 사용해도 괜찮을 것이라고 판단했다.
- 최대값과 최소값을 구하고 최대값에 -1, 최소값에 +1을 횟수 D만큼 수행했다. 이 때 max와 min의 차이가 1 또는 0이라면 아무리 바꿔도 같은 결과가 나오기 때문에 중단시켜주었다.