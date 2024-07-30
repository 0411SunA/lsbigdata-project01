import pandas as pd
import numpy as np

old_seat = np.arange(1, 29)
np.random.seed(20240729)
new_seat = np.random.choice(old_seat, 28, replace = False)

result=pd.DataFrame(
    {"old_seat": old_seat,
     "new_seat": new_seat}
)

result.to_csv(result, "result.csv")

# y=2x 그래프 그리기
# 점을 직선으로 이어서 표현
x = np.linspace(0, 8, 2)
y = 2*x

#plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.show()
plt.clf()

# y = x^2 점 세개를 사용해서 그려보아라.
x = np.linspace(-8, 8, 100)
y = x ** 2

# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")

# x, y축 범위
plt.xlim(-10, 10)
plt.ylim(0, 40)
# plt.axis('ecual')는 xlim, ylim과 같이 사용 x
plt.gca().set_aspect('equal', adjustable='box')

# 비율 맞추기
# plt.axis('equal')
                                                 
plt.show()
plt.clf()

# 작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다. 이 정보를 이번 년도 남학생
# 분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.
# 실제 정답!
import numpy as np

# 표본 평균 x_ = 68.9
# n = 16
# sigma= 6
# alpha: 0.1 (1-alpha: 0.9(신뢰수준))
# z: a/2 = z 0.05 뜻: 정규분포(mu = 0, sigma**2 = 1은 특히 표준정규분포!)에서, 상위 5%에 해당하는 x값. 
# 파이썬에서 norm.ppf(0.95, loc=0, scale=1)
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean()
len(x)
z_005 = norm.ppf(0.95, loc=0, scale=1)
z_005
#신뢰구간
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)

# 데이터로부터 E[X^2] 구하기
x = norm.rvs(loc=3, scale=5, size=100000)

# 분산 구하기 두가지 방법
np.mean(x**2)
sum(x**2) / (len(x) - 1)

# E[(X-X**2)/(2X)]
np.mean((x - x**2) / (2*x))

#위 기법은 몬테카를로 적분. 확률변수 기대값을 구할 때, 표본을 많이 뽑은 후, 원하는 
# 형태로 변형, 평균을 계산해서 기대값을 구하는 방법

# 표본 10만개 추출해서 s**2 구하세요
np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size=100000)
# 첫번째 x_bar 구해야함. 표본평균
x_bar = x.mean()
x_bar
s_2 = sum((x - x_bar) **2) / (100000-1)
s_2
# 쉬운 방법
# np.var(x) 사용하면 안됨 주의! # n으로 나눈 값
np.var(x, ddof=1) # n-1로 나눈 값 (표본 분산)

# n-1 vs n.
x = norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)














