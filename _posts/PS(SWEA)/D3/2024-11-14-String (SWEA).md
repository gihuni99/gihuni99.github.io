---
title: 1213. String (SWEA)
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

[SW Expert Academy](https://swexpertacademy.com/main/code/problem/problemDetail.do?problemLevel=3&contestProbId=AV14P0c6AAUCFAYi&categoryId=AV14P0c6AAUCFAYi&categoryType=CODE&problemTitle=&orderBy=INQUERY_COUNT&selectCodeLang=PYTHON&select-1=3&pageSize=10&pageIndex=2)

# 나의 풀이

```python
for test_case in range(1, 11):
    T = int(input())
    s=input()
    string=input()
    si=0
    count=0
    for ss in string:
        if ss==s[si]:
            si+=1
        else:
            si=0
            if ss==s[si]:
                si=1
        if si==len(s):
            count+=1
            si=0
    print(f'#{test_case} {count}')
```

- https://devjhs.tistory.com/683
- count()를 사용하면 간단하지만, 연습이기 때문에 구현을 해보았다.
- 일치하는 문자가 나오면 counting을 하고, 모든 문자열이 일치하면 다시 0으로 만든 뒤 답에 +1을 해준다.
- 이때 순서에 따른 문자열이 일치하지 않을 때, 해당 문자가 가장 첫번째와 일치하는지 여부를 판단해주어야 한다.