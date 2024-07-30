# 이항분포 베르누이 확률변수 binom.pmf((x가 나올 확률), n=, p=)

# 이항분포 X ~ P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
from scipy.stats import binom

binom.pmf(0, n = 2, p=0.3)
binom.pmf(1, n = 2, p=0.3)
binom.pmf(2, n = 2, p=0.3)                               

result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)

# 팩토리얼 문제
import math
math.factorial(54) / (math.factorial(26) * math.factorial(54-26))
math.comb(54,26)

math.comb(2, 0) * 0.3 ** 0 * (1-0.3) ** 2
binom.pmf(0, 2, 0.3)

binom.pmf(4, n=10, p=0.36)

np.arange(5)
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# X~B(30, 0.2)
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
b = binom.pmf(np.arange(25,31), n=30, p=0.2).sum()
a + b
!pip install scipy
from scipy.stats import bernoulli

bernoulli.rvs(p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)


binom.rvs(n=2, p=0.3, size=1)

binom.pmf(0, n=2, p=0.3)

binom.rvs(n = 30, p = 0.26, size = 30)

#문제. X~B(30, 0.26) 시각화 해보세요! -> 막대그래프

plt.clf()
import pandas as pd
import seaborn as sns

x = np.arange(31)
prob_x = binom.pmf(np.arange(31), n = 30, p =0.26)

sns.barplot(prob_x)
import matplotlib.pyplot as plt
plt.show()
plt.clf()

df = pd.DataFrame({"x":x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()

binom.cdf(4, n=30, p=0.26)

a = binom.cdf(18, n=30, p=0.26)
b = binom.cdf(4, n=30, p=0.26)
a - b

# size: 공 개수 설정하기
# 1이 나올 확률이 p인 함수를 30개 뽑는다. size는 총 게임 횟수라고 생각하기
x_1 = binom.rvs(n = 30, p=0.26, size = 10)
x_1 

x = np.arange(31)
prob_x = 
















