from scipy.stats import norm
# X ~ N(3, 7^2) 하위 25%에 해당하는 X는
x = norm.ppf(0.25, loc=3, scale=7)
# Z~N(0, 1) 에서 하위 25%에 해당하는 Z는
z = norm.ppf(0.25, loc=0, scale=1)
z
z*7 + 3
# x = 3 + z * 7
# x-3/7 = z
# x-mu/sigma = z

# X~N(3, 7^2)
# N(0, 1^2)에서 2/7보다 작을 확률
norm.cdf(5, loc = 3, scale=7)
norm.cdf(2/7, loc = 0, scale=1)

norm.ppf(0.975, loc=0, scale=1)

# <표준정규분포> 표본 1000개, 히스토그램 -> pdf 겹쳐서 그리기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import uniform

z=norm.rvs(loc=0, scale=1, size=1000)
z

x=z*np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="pink")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

# 정규분포 X~N(5, 3^2)
# Z = X-5/3 가 표준정규분포를 따르나요?
# X에세 3 빼고 5로 나눠서 표준정규분포를 그린다.
# X 표본 1000개 뽑음 -> 표준화 사용 X를 Z로 변형 ->
# Z의 히스토그램 그리기 -> 표준정규분포 pdf 겹치기
x=norm.rvs(loc=5, scale=3, size=1000)

z=(x-5) /3 #표준화
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=5, scale=np.sqrt(3))
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# 표본표준편차 나눠도 표준정규분포가 될까? 
# 1. X 표본을 10개를 뽑아서 표본분산값 계산.
# 2. X 표본 1000개 뽑음
# 3. 1번에서 계산한 s^2으로 sigma^2 대체한 표준화를 진행. 
# z = x - mu/sigma 대신 sigma자리를 s로 대체해라.
# 4. Z의 히스토그램 그리기 -> 표준정규분포 pdf 확인.

# 정규분포 X ~ N(5, 3^2)
# 1. X 표본을 10개를 뽑아서 표본분산값 계산.
x=norm.rvs(loc=5, scale=3, size=10)
s = np.std(x, ddof=1)
#s_2 = np.std(x, ddof=1)

# 2. X 표본 1000개 뽑음
x=norm.rvs(loc=5, scale=3, size=1000)

# 표준화
z=(x-5) /s
sns.histplot(z, stat="density", color="grey")
# 3. 1번에서 계산한 s^2으로 sigma^2 대체한 표준화를 진행. 

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# t 분포에 대해서 알아보자!
# x ~ t(df) -> 모수가 하나짜리임.
# 종모양, 대칭분포, 중심 0으로 정해져 있음.
# 모수 df: 자유도라고 부름 - 퍼짐을 나타내는 모수
# n이 작으면 분산 커짐. 
from scipy.stats import t
# t.pdf
# t.ppf
# t.cdf
# t.rvs
# 위 4개 다 사용 가능.
# 자유도가 4인 t분포의 pdf를 그려보세요!
# 자유도가 커질수록 표준정규분포와 유사해짐.
# n이 무한대로 가면 표준정규분포가 된다.
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df = 30)
plt.plot(t_values, pdf_values, color='red', linewidth=2)

#표준정규분포 겹치기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()

# 모평균의 신뢰구간을 구할 수 있다. 
# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도가 n-1인 t 분포
x = norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
x_bar = x.mean()
n = len(x)
#모평균에 대한 95% 신뢰구간을 구해보자
# 모분산을 모를때
x_bar + t.ppf(0.975, df = n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df = n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar

# 모분산을 알 때
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 /  np.sqrt(n)










































