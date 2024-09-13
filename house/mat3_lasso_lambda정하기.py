import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso
# 결과 받기 위한 벡터 만들기 위해
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

i=1
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

# 잔차제곱합
    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]
# y축 아래로 갈수록 성능 좋음(매장)
# y축 위로 가면 성능 안좋음

model = Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)
sorted_train = train_x.sort_values("x")
reg_line = model.predict(sorted_train)

plt.plot(sorted_train["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"])
#----------------------------
# 라쏘 얼마나 적합한지 알아보자
# 추정된 라쏘(lambda=0.03) 모델을 사용해서, -4, 4까지 간격 0.01 x에 대해 예측값을 계산,
# 산점도에 valid set 그린 다음, -4, 4까지 예측값을 빨간 선으로 겹쳐서 그릴것
from scipy.stats import norm
from scipy.stats import uniform

# 검정 곡선
k = np.linspace(-4, 4, 801)
sin_y = np.sin(k)

# 파란 점들
x = uniform.rvs(size=20, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=20, loc=0, scale=0.3)

# plt.plot(k, sin_y, color="black")
plt.scatter(x, y, color="blue")

model = Lasso(alpha = 0.03)
model.fit(train_x, train_y)

df_k = pd.DataFrame({
    "x": k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8,
    "x9": k**9, "x10": k**10, "x11": k**11, "x12": k**12, "x13": k**13,
    "x14": k**14, "x15": k**15, "x16": k**16, "x17": k**17, "x18": k**18,
    "x19": k**19, "x20": k**20
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")
#-----정답----------------------
k=np.linspace(-4, 4, 801)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i
    
k_df

reg_line = model.predict(k_df)

plt.plot(k_df["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")
#------------------- 또 다른 답
# lasso회귀 valid set에 얼마나 적합하는지 보자
k = np.arange(-4, 4, 0.01)
df_k = pd.DataFrame({
    'x': k
    })
for i in range(2,21) :
    df_k[f'x{i}'] = df_k['x']**i
plt.scatter(valid_df['x'], valid_df['y'], color='blue')
reg_line = model.predict(df_k)
plt.plot(k, reg_line, color="red")
plt.show()
#------------------------------------
# seaborn을 사용하여 산점도 그리기
# 무작위로 뽑아서 그룹 나눠도 되는데 해도되고 안해도 됨. 6개씩 그룹 나눠서
# valid set에서 평균과 분산을 알아보자 람다값이 0.3일때 적절하다고 판단했으므로 진짜인지 그룹 나눠서
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

# train 셋을 5개로 쪼개어 valid set과 train set을 5개로 만들기
# 각 세트에 대한 성능을 각 lambda 값에 대응하여 구하기
# 성능 평가 지표 5개를 평균내어 오른쪽 그래프 다시 그리기
# seaborn을 사용하여 산점도 그리기
# 모의고사 5개 보기
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df
np.random.seed(42) 
index = np.random.choice(30, 30, replace = False)

a_1 = index[:6]
a_2 = index[6:12]
a_3 = index[12:18]
a_4 = index[18:24]
a_5 = index[24:30]

train_df1 = df.drop(a_1)
train_df2 = df.drop(a_2)
train_df3 = df.drop(a_3)
train_df4 = df.drop(a_4)
train_df5 = df.drop(a_5)

train_df1_x = train_df1[['x']]
train_df2_x = train_df2[['x']]
train_df3_x = train_df3[['x']]
train_df4_x = train_df4[['x']]
train_df5_x = train_df5[['x']]

train_df1_y = train_df1['y']
train_df2_y = train_df2['y']
train_df3_y = train_df3['y']
train_df4_y = train_df4['y']
train_df5_y = train_df5['y']

valid_df1 = df.iloc[a_1,:]
valid_df2 = df.iloc[a_2,:]
valid_df3 = df.iloc[a_3,:]
valid_df4 = df.iloc[a_4,:]
valid_df5 = df.iloc[a_5,:]

valid_df1_x = valid_df1[['x']]
valid_df2_x = valid_df2[['x']]
valid_df3_x = valid_df3[['x']]
valid_df4_x = valid_df4[['x']]
valid_df5_x = valid_df5[['x']]

valid_df1_y = valid_df1['y']
valid_df2_y = valid_df2['y']
valid_df3_y = valid_df3['y']
valid_df4_y = valid_df4['y']
valid_df5_y = valid_df5['y']

# 결과 받기 위한 벡터 만들기
val1_result=np.repeat(0.0, 100)
tr1_result=np.repeat(0.0, 100)

# fold1
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df1_x, train_df1_y)

    # 모델 성능
    y_hat_train = model.predict(train_df1_x)
    y_hat_val = model.predict(valid_df1_x)

    perf_train=sum((train_df1["y"] - y_hat_train)**2)
    perf_val=sum((valid_df1["y"] - y_hat_val)**2)
    tr1_result[i]=perf_train
    val1_result[i]=perf_val

val1_result=np.repeat(0.0, 100)
tr1_result=np.repeat(0.0, 100)

#fold2
val2_result=np.repeat(0.0, 100)
tr2_result=np.repeat(0.0, 100)
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df2_x, train_df2_y)

    # 모델 성능
    y_hat_train = model.predict(train_df2_x)
    y_hat_val = model.predict(valid_df2_x)

    perf_train=sum((train_df2["y"] - y_hat_train)**2)
    perf_val=sum((valid_df2["y"] - y_hat_val)**2)
    tr2_result[i]=perf_train
    val2_result[i]=perf_val

#fold3
val3_result=np.repeat(0.0, 100)
tr3_result=np.repeat(0.0, 100)
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df3_x, train_df3_y)

    # 모델 성능
    y_hat_train = model.predict(train_df3_x)
    y_hat_val = model.predict(valid_df3_x)

    perf_train=sum((train_df3["y"] - y_hat_train)**2)
    perf_val=sum((valid_df3["y"] - y_hat_val)**2)
    tr3_result[i]=perf_train
    val3_result[i]=perf_val

#fold4
val4_result=np.repeat(0.0, 100)
tr4_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df4_x, train_df4_y)

    # 모델 성능
    y_hat_train = model.predict(train_df4_x)
    y_hat_val = model.predict(valid_df4_x)

    perf_train=sum((train_df4["y"] - y_hat_train)**2)
    perf_val=sum((valid_df4["y"] - y_hat_val)**2)
    tr4_result[i]=perf_train
    val4_result[i]=perf_val
#fold 5
val5_result=np.repeat(0.0, 100)
tr5_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df5_x, train_df5_y)

    # 모델 성능
    y_hat_train = model.predict(train_df5_x)
    y_hat_val = model.predict(valid_df5_x)

    perf_train=sum((train_df1["y"] - y_hat_train)**2)
    perf_val=sum((valid_df5["y"] - y_hat_val)**2)
    tr5_result[i]=perf_train
    val5_result[i]=perf_val

tr_result = (tr1_result + tr2_result + tr3_result + tr4_result + tr5_result) / 5
val_result = (val1_result + val2_result + val3_result + val4_result + val5_result) / 5


df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)


val_result.argmin()