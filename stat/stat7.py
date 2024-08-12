# y=2x+3 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# x 값의 범위 설정
x = np.linspace(0, 100, 400)

# y 값 계산
y = 2 * x + 3

#np.random.seed(20240805)
obs_x=np.random.choice(np.arange(100), 20)
# 줄일 수 없는 잔차들 epsilon이다. 
# 엡실론이 있어야 식 추정 가능
# 노이즈 같은, 예를 들어 레이더에 목표물이 어디잇는지 신호를 받는데
# 그 좌표가 정확하지 않으니까 그 좌표에 대한 오차값 같은..근데 오차는 아닌
# scale 작을수록 빨간 선이랑 비슷해진다. 
# 즉 분산이 작을수록 원래 발생시키는 직선이랑 비슷하게 나올 것 같네

epsilon_i=norm.rvs(loc=0, scale=10, size=20)
obs_y = 2*obs_x+3 + epsilon_i

# 그래프 그리기
plt.plot(x, y, label='y = 2x + 3', color="black")
plt.scatter(obs_x, obs_y, color="blue", s=3)
# plt.show()
# plt.clf()

import pandas as pd
df = pd.DataFrame({"x": obs_x,
                   "y": obs_y})
df                   

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

#모델 학습
obs_x = obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a hat
model.intercept_ # 절편 b hat

# 회귀 직선 그리기
x = np.linspace(0, 100, 400)
# coef_[0] numpy로 가져오고 싶어서
y = model.coef_[0] * x + model.intercept_
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.plot(x, y, color="red") # 회귀직선
plt.show()
plt.clf()

model
summary(model)

# ! pip install statsmodels                                                                                                                                         
import statsmodels.api as sm

model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())

# -----------오후 수업-------------
# 테스트의 개념, 너의 말이 맞냐 안맞냐 확인하는 과정.
# 통계검정에서 중요한 것: 유의수준, 유의확률
# 유의수준 = 기각역에 대응하는 확률값, 참일 때 분포가 중심극한 분포가 된다. 수준 넘어서면 기각됨. 
# 기각역은 알파, 
# 신뢰구간은 1 - (norm.ppf(18, loc=16, scale=1.96))* 2 양측검정이어서 2 곱하기

# 14~16에서 돈 벎. 18에서 떨어질 확률을 구하시오. cdf 사용하게)
# 귀무가설 가정하에 맞다는 조건에서 시뮬레이터 모평균 구하기
norm.cdf(18, loc=10, scale = 1.96)

sigma / spurt 루트앤 # n 임이어었음

# 유의 확률 p-value()

# 교재 p.41

1 - norm.cdf(4.08, loc=0, scale=1)
from scipy.stats import t, norm

# 문제
# 2. 검정을 위한 가설을 명확하게 서술하시오.
# 평균 복합 에너지 소비효율이 16.0 이상일 것이다.
# h_0 = mu0 >= 16.0
# h_1 = mu0 < 16.0

# 3. 검정통계량 계산하시오. 모표준편차 모를때, 표본 크기가 작을 때 사용함. 표준화를!
x = [15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,\
15.382, 16.709, 16.804]
t_value = (np.mean(x) - 16) / (np.std(x, ddof=1) / np.sqrt(len(x)))
round(t_value, 3)

# 4. p‑value을 구하세요.
mean_x = np.mean(x)
std_x = np.std(x, ddof=1)
n = len(x)
t.cdf(t_value, df=14) * 2

# 6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.
import numpy as np

ci_95 = t.interval(0.95, df=n-1, loc=mean_x,\
scale=std_x/np.sqrt(n))
print("모평균에 대한 95% 신뢰구간: ", np.round(ci_95, 2))

# 7. 평균 복합에너지 소비효율이 15이라면, 위의 검정의 검정력을 구하세요. (단, 검정력 계산시,
# 모분포의 표준편차를 1라고 가정한다.)
ci_95 = t.interval(0.95, df= n-1, loc=15,\
scale=std_x/np.sqrt(n))
print("모평균에 대한 95% 신뢰구간: ", np.round(ci_95, 2))


ci_95 = t.interval(0.95, df= n-1, loc=15,\
scale=1)
print("모평균에 대한 95% 신뢰구간: ", np.round(ci_95, 2))

#---------
# 재희 언니
n = 15
mu0 = 16

#표본평균
x_mean = x.mean(x)

#표본표준편차 s
x_std = np.std(x, ddof=1)

# t 계산하기 - 검정통계량
t_value = (x_mean - 16)/(x_std / np.sqrt(n))

#유의확률 p-value 구하기
t.cdf(t_value, df = n-1) # 0.04%

# 유의수준 1%와 비교해서 기각 결정
0.01 < 0.04

#모분산을 모를 때: 모평균에 대한 95% 신뢰구간 구하자
#신뢰구간 계산: 평균 +_ (z-값 * SE)
# SE = std/sqrt(n)
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)








