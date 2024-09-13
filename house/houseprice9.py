import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
# 필요한 패키지 불러오기

# 현재 작업 디렉토리 확인
print(os.getcwd())

# 파일이 있는 디렉토리로 변경
os.chdir("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house/train.csv")
house_test=pd.read_csv("./data/house/test.csv")
sub_df=pd.read_csv("./data/house/sample_submission.csv")

## NaN 채우기
## test 셋 결측치 채우기 train과 test는 엄격하게 분리되어있음. 
# train에서 해줬던 걸 test에서 똑같이 해줘야함. 
house_train.isna().sum()
house_test.isna().sum()

# 각 숫자변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
# 가장 기본적인 코드가 됨.
# 숫자형 뽑아와서 8개 결측치 있는 거 확인하고 뚫려있는 애들만 true 나오도록.
# quantitative: 수치형
house_train = house_train.iloc[:, 1:] # id drop 
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected: # 각각 컬럼에 대해 fillna 평균으로 채워라
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

# 범주형 채우기
quantitative = house_train.select_dtypes(include = [object])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected: # 각각 컬럼에 대해 fillna 평균으로 채워라
    house_train[col].fillna("unknown", inplace=True)
house_train[quant_selected].isna().sum()

## 이상치 탐색 (여기 넣으면 안됨!)
# house_train=house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
#house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)