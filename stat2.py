import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)
# 데이터 개수 확인하기
sum(data < 0.18)
# 히스토그램 그리기

# bins : 높이구간, 
# alpha: 투명도
plt.hist(data, bins=4, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
# grid : 선
plt.grid(False)
plt.show()

plt.clf()

# rand. -> 파란색 함수 평균 엄청 많이 만들기 -> 종모양 히스토그램 나옴
# 0~1 사이 숫자 5개 발생
# 2. 표본 평균 계산하기
# 1,2단계를 10000번 반복한 결과를 벡터로 만들기
# 히스토그램 그리기
# hint: numpy 행렬
#1. 
# data = np.random.rand(5)
# x_bar = data.mean()
# x_bar
# #2.
# import numpy as np
# def X(i):
#     np.random.rand(5)
#     x_bar = data.mean()
#     x_bar
#     return np.random.rand(i)
# X(10000)
# 정답

# np.random.rand(10000,5).reshape(-1, 5).mean(axis=1)
plt.clf()
# 0~1 사이 숫자 5개 발생
# 2. 표본 평균 계산하기
# 1,2단계를 10000번 반복한 결과를 벡터로 만들기
# 히스토그램 그리기

x=np.random.rand(50000)\
    .reshape(-1, 5)\
    .mean(axis=1)
plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
# grid : 선
plt.grid(False)
plt.show()
plt.clf()

x = np.random.rand(50000) \
      .reshape(-1, 5)\
      .mean(axis=1)
plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
# grid : 선
plt.grid(False)
plt.show()
plt.clf()

#기댓값 16 유도하기
# np.arange(33) -> 확률변수 x가 가질 수 있는 값
x = np.arange(33)
x.sum() / 33
np.arange(257).sum() / 33

(np.arange(33) - 16)**2
# 33분의 2가 됨. 0은 유일하게 1/33
np.unique((np.arange(33) - 16)**2)
#분산 구하기
sum(np.unique((np.arange(33) - 16) ** 2) * (2/33))

# E[X^2]
sum(x ** 2 * (1/33))

# Var(X) = E[X^2] - (E[X])^2
sum(x ** 2 * (1/33)) - 16**2

(114-81)/36

# 예제
x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x.sum()
pro_x

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)
# 분산
Exx - Ex**2

sum((x - Ex)**2 * pro_x)

# 예제. x = 0~98까지 정수, 
# 1, 2, 3, ..., 49, 50... 
# 1/2500

x = np.arange(99)
x

# 1-50-1 벡터
x_1_50_1 = np.concatenate((np.arange(1, 51),np.arange(49, 0, -1))
pro_x = x_1_50_1/2500
pro_x = np.array()
pro_x

#기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

#분산
Exx - Ex**2
sum((x-Ex)**2 ( pro_x))
sum(np.arange(50)) + 
# 문제 2 y = 0,2,4,6
x = np.arange(0,4) * 2 
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x.sum()
pro_x

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)
# 분산
Exx - Ex**2

sum((x - Ex)**2 * pro_x)
# Var(2x) = 2^2Var(x)
# Var(2x)      = 4 * 0.916

# n = 16 표준편차 계산해봐라
np.sqrt(9.52**2/16)
# n = 10 표준편차
np.sqrt(9.52 **2/10)


























