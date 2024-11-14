---
title: 2806. N-Queen (SWEA)
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

**※ SW Expert 아카데미의 문제를 무단 복제하는 것을 금지합니다.**

출처: 

[SW Expert Academy](https://swexpertacademy.com/main/code/problem/problemDetail.do?problemLevel=3&contestProbId=AV7GKs06AU0DFAXB&categoryId=AV7GKs06AU0DFAXB&categoryType=CODE&problemTitle=&orderBy=RECOMMEND_COUNT&selectCodeLang=PYTHON&select-1=3&pageSize=10&pageIndex=1)

# 나의 풀이

```python
def check(current,b):
    r,c=current
    for i in range(1,r+1):
        if b[r-i][c]:
            return False
        if 0<=c-i<len(b) and b[r-i][c-i]:
            return False
        if 0<=c+i<len(b) and b[r-i][c+i]:
            return False
    return True

def dfs(s,b):
    global cnt
    if s==N:
        cnt+=1
        return
    for i in range(N):
        if check([s,i],b):
            b[s][i]=1
            dfs(s+1,b)
            b[s][i]=0

T = int(input())
for test_case in range(1, T + 1):
    N=int(input())
    cnt=0
    b=[[0]*N for _ in range(N)]
    dfs(0,b)
    print(f'#{test_case} {cnt}')
```

- dfs로 구현했다.
- 하나의 행에는 하나의 말 밖에 올 수 없다. 따라서 한 행에는 총 N가지의 경우의 수가 나온다.
- 이전 행에 같은 열 또는 대각선에 말이 있다면 check함수로 False를 반환하여 더 이상 탐색하지 않도록 하였다.
- 최종 N에 도달한 수를 count하여 답으로 출력하였다.