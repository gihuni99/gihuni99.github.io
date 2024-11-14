---
title: 1220. Magnetic (SWEA)
date: 2024-11-14 00:00:00 +09:00
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

출처: 

[SW Expert Academy](https://swexpertacademy.com/main/code/problem/problemDetail.do?problemLevel=3&contestProbId=AV14hwZqABsCFAYD&categoryId=AV14hwZqABsCFAYD&categoryType=CODE&problemTitle=&orderBy=INQUERY_COUNT&selectCodeLang=PYTHON&select-1=3&pageSize=10&pageIndex=1)

# 나의 풀이

```python
T = 10
for test_case in range(1, T + 1):
    table=[]
    cnt=0
    table_size=int(input())
    for _ in range(table_size):
        table.append(list(map(int,input().strip().split())))
    for i in range(table_size):
        N_ex=False
        for j in range(table_size):
            if table[j][i]==1:
                N_ex=True
            elif table[j][i]==2 and N_ex:
                cnt+=1
                N_ex=False
    print(f'#{test_case} {cnt}')
```

- 각 column별로 N극이 먼저 있는지 확인한다. N극이 있는 상태에서 다음에 S극이 나온다면 교착 상태 하나가 추가된다.
- 교착 상태가 하나 만들어지면 다시 N극을 찾고, 그 다음 S극을 다시 찾는다.