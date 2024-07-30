# 균일확률변수 만들기

import numpy as np
np.random.rand(1)

def X(i):
    return np.random.rand(i)

X(1)    
X(2)    
X(3)    

# 각각 독립적으로 시행됨.
# return np.random.rand(i) -> 중복적으로 나올 수도 있음. 독립시행

# 베르누이 확률변수 모수: p 만들어보세요!
# num = 3
# p = 0.5

def Y(num , p):
    x = np.random.rand(num)
    return np.where(x < p,1,0) 

# 평균
# sum(Y(num = 100, p = 0.5)) / 100
Y(num=1, p=0.5)
Y(num = 100, p = 0.5).mean()
Y(num = 100000, p = 0.5).mean()

# 문제
# 새로운 확률변수
# 가질 수 있는 값: 0, 1, 2
# 20%, 50%, 30%

# 오류나는 코드
# def Z():
#     x=np.random.rand(1)
#     return np.where(x < 0.2, 0, np.where(x < 0.7, 1, 2))
# 
# Z()

# p = np.array([0.2, 0.5, 0.3)
def Z(p):
    x=np.random.rand(1)
    p_cumsum=p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

p = np.array([0.2, 0.5, 0.3])
Z(p)

# 0719
# E[X] 확률분포표
# X = [0,1,2,3]
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)

#베르누이 함수 분포
def Y(num, p):
    x = np.random(num)
    return np.where(x < p,1,0) 


