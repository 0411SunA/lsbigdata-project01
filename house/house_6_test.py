# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/train.csv")
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/test.csv")
sub_df = pd.read_csv('./data/house/sample_submission.csv')


# 수치형 변수인 int와 float만 빼고 싶음 
house_train.info()

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include = [int, float])
x.info()
# 필요없는 칼럼 제거하기
x.iloc[:,1:-1]
y = house_train["SalePrice"]
x.isna().sum()
house_train['LotFrontage'].describe()
house_train['MasVnrArea'].describe()
house_train['GarageYrBlt'].describe()
house_train[['LotFrontage','MasVnrArea','GarageYrBlt' ]].describe()

house_train['LotFrontage'] = house_train['LotFrontage'].fillna(house_train['LotFrontage'].mean())
house_train['GarageYrBlt'] = house_train['GarageYrBlt'].fillna(house_train['GarageYrBlt'].mean())
house_train['MasVnrArea'] = house_train['GarageYrBlt'].fillna(0)
house_train.isna()
house_train.info()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# # test에 넣기
# test_x = house_test.select_dtypes(include = [int, float])
# test_x
# test_x.isna().sum()
# test_x[["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", \
# "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",\
# "BsmtHalfBath", "GarageYrBlt", "GarageCars",\
# "GarageArea"]].describe()
# 
# test_x['BsmtFinSF1'] = test_x['BsmtFinSF1'].fillna(0)
# test_x['BsmtFinSF2'] = test_x['BsmtFinSF2'].fillna(0)
# test_x['BsmtUnfSF'] = test_x['BsmtUnfSF'].fillna(0)
# test_x['TotalBsmtSF'] = test_x['TotalBsmtSF'].fillna(0)
# test_x['GarageCars'] = test_x['GarageCars'].fillna(0)
# test_x['GarageArea'] = test_x['GarageArea'].fillna(0)
# test_x['BsmtFullBath'] = test_x['BsmtFullBath'].fillna(0)
# test_x['BsmtHalfBath'] = test_x['BsmtHalfBath'].fillna(0)
# 
# test_x[["LotFrontage", "MasVnrArea", "GarageYrBlt"]].describe()
# test_x['LotFrontage'] = test_x['LotFrontage'].fillna(test_x['LotFrontage'].mean())
# test_x['MasVnrArea'] = test_x['MasVnrArea'].fillna(0)
# test_x['GarageYrBlt'] = test_x['GarageYrBlt'].fillna(test_x['GarageYrBlt'].mean())
# 
# test_x.isna().sum()
# 
# # 결측치 확인
# test_x["GrLivArea"].isna().sum()
# test_x["GarageArea"].isna().sum()
# test_x=test_x.fillna(house_test["GarageArea"].mean())
# 
# # 테스트 데이터 집값 예측
# pred_y=model.predict(test_x) # test 셋에 대한 집값
# pred_y
# 
# # SalePrice 바꿔치기
# sub_df["SalePrice"] = pred_y
# sub_df
# 이상치 탐색 및 제거
quantitative = house_train.select_dtypes(include = ['number'])
quantitative.info()
x = quantitative.iloc[:, 1:-1]
x.isna().sum()

fill_values = {
    'LotFrontage': x["LotFrontage"].mean(),  
    'MasVnrArea': x["MasVnrArea"].mean()[0], 
    'GarageYrBlt': x["GarageYrBlt"].mean()
}
x = x.fillna(fill_values)
y = quantitative.iloc[:, -1]
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
slope = model.coef_      # 기울기 a
intercept = model.intercept_ # 절편 b
y_pred = model.predict(x)

quantitative1 = house_test.select_dtypes(include = ['number'])
test_x = quantitative.iloc[:, 1:-1]
test_x.isna().sum()

fill_values = {
    'LotFrontage': x["LotFrontage"].mean(),  
    'MasVnrArea': x["MasVnrArea"].mean()[0], 
    'GarageYrBlt': x["GarageYrBlt"].mean()
}

test_x = test_x.fillna(fill_values)
test_x = test_x.fillna(0)

y_hat = moodel.predict(test_x)
sub_df["SalePrice"] = y_hat
sub_df.to_csv("./data/house/sample_submission8.csv", index=False)



#-----------------------수업시간-----------------
# 변수별로 결측값 채우기
#mode()[0]으로 해야함. mode는!
fill_values = {
    'LotFrontage' : x["LotFrontage"].mean(),
    'MasVnrArea'  : x['MasVnrArea'].mean(),
    'GarageYrBlt' : x['GarageYrBlt'].mean()
}
x_filled = x.fillna(value=fill_values)

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include= [int, float])
test_x = test_x.iloc[:, 1:]

fill_values = {
    'LotFrontage' : test_x["LotFrontage"].mean(),
    'MasVnrArea'  : test_x['MasVnrArea'].mean(),
    'GarageYrBlt' : test_x['GarageYrBlt'].mean()
}
test_x = test_x.fillna(value = fill_values)
test_x = test_x.fillna(test_x.mean())
test_x.isna().sum()

#-----------수업-------------------------------
# csv 파일로 내보내기
sub_df.to_csv("./data/house/sample_submission8.csv", index=False)
# 시각화
# 직선값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()
