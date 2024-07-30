!pip install scipy
from scipy.stats import bernoulli

#확률질량함수(pmf)
#확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
#bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# 이항분포 X ~ P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
from scipy.stats import binom

# 베르누이 확률변수 n개 더한 것 구할 수 있구나
# 각각의 확률변수의 p 값이 동일해야함.
binom.pmf(0, n = 2, p=0.3)
binom.pmf(1, n = 2, p=0.3)
binom.pmf(2, n = 2, p=0.3)

# X ~ B(n, p)
# x+ = x1+...+x30 = 0,1,2,...30 p=0.3
# lisc comp.
result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

#numpy
import numpy as np
#30까지 넣어서 계산해라.
binom.pmf(np.arange(31), n=30, p=0.3)

# 팩토리얼 문제
#(54)
#(26)
import math
math.factorial(54) / (math.factorial(26) * math.factorial(54-26))
math.comb(54,26)


# np로 계산
# 이 방법 안됨//
fact_54 = np.cumprod(np.arange(1, 55))[-1]
np.cumprod(np.arange(1, 26))[-1]
np.cumprod(np.arange(1, 28))[-1]

#다른 방법은 가능
# 힌트: ln
log(a * b) = log(a) + log(b)
# ==========몰라도됨=========
log(1 * 2 * 3* 4) = log(1) + log(2) + log(3) + log(4)
math.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial((54)))
sum(np.log(np.arange(1, 55)))
# 문제
logf_54 = sum(np.log(np.arange(1, 55))) 
logf_26 = sum(np.log(np.arange(1, 27))) 
logf_28 = sum(np.log(np.arange(1, 29))) 
# math.comb(54,26)
np.exp(logf_54 - (logf_26 + logf_28))
# ===============================

# 2C0 0.3^0(1-0.3)^2
math.comb(2, 0) * 0.3**0 * (1-0.3) ** 2
# 2C1 0.3^1(1-0.3)^1
math.comb(2, 1) * 0.3**1 * (1-0.3) ** 1
# 2C2 0.3^2(1-0.3)^0 
math.comb(2, 2) * 0.3**2 * (1-0.3) ** 0

# pmf: probability mass function (확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X~B(n=10, p=0.36)
# P(X=4) = ?
binom.pmf(4, n=10, p=0.36)

# P(X<=4) = ?
np.arange(5)
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# P(2<X<=8) = ?
#내 답
np.arange(3)
a = binom.pmf(np.arange(3), n=10, p=0.36).sum()
b = binom.pmf(np.arange(9), n=10, p=0.36).sum()
b - a

#정답
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()

# X~B(30, 0.2)
# 확률변수 X가 4보다 작거나, 25보다 크거나 같을 확률을 구하시오.
# P(X<4 or X>=25)
X~B(30, 0.2)
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
b = binom.pmf(np.arange(25,31), n=30, p=0.2).sum()
a + b
# 1. P(X<4)
# 2. P(X>=25)
# 3. 1+2
# 다른 방법: 1 - P(4<=X<=25)
c = binom.pmf(np.arange(4,25), n=30, p=0.2).sum()
1 - c

# rvs 함수 (random variates sample)
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
# 한줄로 표현하기 같은 분포 따르는 표본을 여러개 한번에 뽑고싶을 때 사용.
binom.rvs(n=2, p=0.3, size=1)

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X ~ B(30, 0.26)
# E[X] = np
# 표본 30개를 뽑아보세요!
binom.rvs(n=30, p=0.26, size=30)
# E[X]=?
# 내 정답
# 베르누이 확률분포 기대값은 p구나.! 
# E[X] = E[Y1+Y2+...+Y30]
# 정답
30 * 0.26

# E[X] = np -> 외워라!

#문제. X~B(30, 0.26) 시각화 해보세요! -> 막대그래프
plt.clf()
import pandas as pd
import seaborn as sns

x = np.arange(31)
prob_x = binom.pmf(np.arange(31), n=30, p=0.26)

sns.barplot(prob_x)
import matplotlib.pyplot as plt
plt.show()
plt.clf()

#교재 p.207
df = pd.DataFrame({"x":x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()

# cdf: cumulative dist. function
# (누적확률분포 함수)
# F_(X=x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)

# P(4<X<=18) = ?
# P(X<=18) - P(X<=4)
a = binom.cdf(18, n=30, p=0.26)
b = binom.cdf(4, n=30, p=0.26)
a - b

# 연습2. P(13 < X < 20) =?
# = P(X<=19) - P(X<=13)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

import numpy as np
import seaborn as sns

# size: 공 개수 설정하기
# 1이 나올 확률이 p인 함수를 30개 뽑는다. size는 총 게임 횟수라고 생각하기
x_1 = binom.rvs(n = 30, p = 0.26, size = 10)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="pink")

# Add a point at (2,0)
plt.scatter(x_1, np.repeat(0.002, 10), color='red', zorder=100, s=5)
plt.axvline(x = 7.8 , color='green', 
            linestyle = '--', linewidth = 2)

plt.show()
plt.clf()

#x가 0이나올확률~8이 나올확률 다 더하면 0.5가 나옴.
binom.ppf(0.5, n=30, p=0.26)

#7까지 딱 된다.!
binom.cdf(7, n=30, p=0.26)

binom.ppf(0.7, n=30, p=0.26)
#8까지의 확률을 더하면 0.62까지 나옴.
binom.cdf(8, n=30, p=0.26)

#ppf는 cdf의 반대 개념이다.

1/np.sqrt(2 * math.pi)
from scipy.stats import norm
norm.pdf(0, loc=0, scale=1)

# 뮤 3 시그마 4 x 5
norm.pdf(5, loc=3, scale=4)

# 정규분포 pdf 그리기
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.scatter(k, y, color="black")
plt.show()
plt.clf()

## mu loc(): 분포의 중심 결정하는 모수
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color="black")
plt.show()
plt.clf()
# sigma (scale): 분포의 퍼짐 결정하는 모수 (표준편차)
# scale이 작다 -> 표준편차가 작다-> 분산이 작다-> 평균 근처에 값이 많이 나온다.
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)
y2 = norm.pdf(k, loc=0, scale=2)
y3 = norm.pdf(k, loc=0, scale=0.5)

plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()

# P(X<=0)
norm.cdf(0, loc=0, scale=1)
# 전체 값을 알기위해 큰 수를 집어 넣음.
norm.cdf(100, loc=0, scale=1)

# P(-2<X<0.54) =?
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# P(X<1 or X>3) = ?
1 - norm.cdf(3, loc=0, scale=1) + norm.cdf(1, loc=0, scale=1)
# X ~ N(3, 5^2)
# P(3<X<5) = ?

# X ~ N(3, 5^2)
# P(3 < X < 5) =? 15.54%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)
# 위 확률변수에서 표본 1000개 뽑아보자!
x = norm.rvs(loc=3, scale=5, size = 1000)
sum((x > 3) & (x < 5)) / 1000

# 평균: 0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
y = norm.rvs(loc=0, scale=1, size=1000)
sum(y<0) / 1000
np.mean(y < 0)

# 숙제 Qmd
# 1. 정규분포 pdf 값을 계산하는 자신만의
# 파이썬 함수를 정의하고,
# 정규분포 mu = 3, sigma = 2의 pdf를 그릴 것.

# 2. 파이썬 scipy 패키지 사용해서 다음과 같은
# 확률을 구하시오
# X ~ N(2, 3^2)
# 1) P(X < 3)
# 2) P(2 < X < 5)
# 3) P(X < 3 or X > 7)

# 3. LS 빅데이터 스쿨 학생들의 중간고사 점수는
# 평균이 30이고, 분산이 4인 정규분포를 따른다.
# 상위 5%에 해당하는 학생의 점수는?

x = norm.rvs(loc = 3, scale=2, size=1000)
x

# 스케일 높이를 딱 맞춰줌
sns.histplot(x, stat="density") # 120: 2.0 축소

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()






























