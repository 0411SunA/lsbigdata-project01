
# <균일분포> X ~ U(a,b)
# loc: a. scale: b-a
from scipy.stats import uniform
# loc: 구간시작점, scale: 구간 길이
import numpy as np
import matplotlib.pyplot as plt
uniform.rvs(loc=2, scale=4, size=1)

# sigma(scale) : 분포의 퍼짐 결정하는 모수(표준편차)
# 정규분포 pdf 그리기

k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)

plt.plot(k, y, color="black")
plt.show()
plt.clf()

#높이 알아내기
uniform.pdf(2, loc=2, scale=4)
uniform.pdf(3, loc=2, scale=4)
uniform.pdf(7, loc=2, scale=4)

# P(X<3.25) = ?
# 범위 내의 직사각형의 넓이 계산해도 똑같음.
uniform.cdf(3.25, loc=2, scale=4)
1.25*0.25
# P(5<X<8.39) =?
uniform.cdf(5, loc=2, scale=4)
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)

#상위 7% 값은?
uniform.ppf(0.93, loc=2, scale=4)

# 표본 20개를 뽑아서 표본평균을 계산해보세요.
# random_state=42 특정 값 고정
x = uniform.rvs(loc=2, scale=4, size=20*1000,
                random_state=42)
x = x.reshape(-1,20)
x.shape
blue_x = x.mean(axis=1)
blue_x 

#히스토그램 그리기
plt.clf()
import seaborn as sns
from scipy.stats import norm
sns.histplot(blue_x, stat="density")
plt.show()

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.3333333/20)
# 분산구하는 함수와 기대값 구하는 함수
uniform.var(loc=2, scale=4)
uniform.expect(loc=2, scale=4)

# Plot the normal distribution PDF
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
# uniform.rvs(loc=0, scale=1, size=None, random_state=None)

# 신뢰구간
from scipy.stats import norm
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, 
                      scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
# 95% 신뢰구간에서 a, b 지점을 찾아라
# 양쪽 끝에 5%씩 있어야하니까 2.5% 뗀거야
4 - norm.ppf(0.025, loc=4, scale= np.sqrt(1.33333/20))
4 - norm.ppf(0.975, loc=4, scale= np.sqrt(1.33333/20))

# 99% -> 0.01 /2 = 0.005
# 4 - norm.ppf(0.005, loc=4, scale= np.sqrt(1.33333/20))
# 4 - norm.ppf(0.995, loc=4, scale= np.sqrt(1.33333/20))

#파란 벽돌 점 찍기
# norm.ppf(0.975, loc=0, scale=1) = 1.96

blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
a = blue_x + 1.96 * np.sqrt(1.3333/20)
b = blue_x - 1.96 * np.sqrt(1.3333/20)
plt.scatter(blue_x, 0.002,
            color='blue', zorder=10, s=10)
plt.axvline(x = a, color="blue",
            linestyle="--", linewidth=2)
plt.axvline(x = b, color="blue",
            linestyle="--", linewidth=2)

# 기대값 표현
plt.axvline(x = 4, color="green",
            linestyle="-", linewidth=2)
plt.show()
plt.clf()

x = uniform.rvs(loc=2, scale=4, size=20*1000,
                random_state=42)
x = x.reshape(-1,20)
x.shape
blue_x = x.mean(axis=1)
blue_x 

# 95% 커버 a,b 표준편차 기준 몇 배를 벌리면 됨??
#  = 3.493, b: 4.506
# 0.506 / 루트 1.3333/20 -> 1.96

