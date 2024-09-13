import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str)) # 문자열로 바꿈
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)
# include_bias=False:  
# Polynomial Features 만들 때, 상수 항(1)을 추가하지 않겠다는 뜻.

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)
# 알파값 설정
lasso = Lasso(alpha=0.01)
# ridge = Ridge(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

#------------------------------------------------------
# 해야 할 일
# 데이터 전처리:

# 숫자형 변수: 결측치는 평균 또는 중간값으로 채우기. 이상치(Outliers)가 있으면 처리하기.
# 범주형 변수: 결측치는 "unknown"으로 채우고, 더미코딩(dummy encoding)으로 숫자형으로 변환.
# 검증 세트 설정:

# K-Fold 교차 검증(K-Fold Cross Validation) 사용해 데이터를 나누고, 모델 성능 평가 준비.
# 라쏘 회귀 모델 생성:

# 다양한 alpha 값에 대해 라쏘 회귀(Lasso Regression) 모델을 학습시켜 최적의 alpha 값을 찾기.
# 최적의 alpha 값을 찾으면, 이를 사용해 최종 모델을 학습.
# 예측 및 제출:

# 최종 모델로 테스트 데이터의 SalePrice 예측.
# 예측 결과를 제출할 CSV 파일로 저장.
# 각 숫자변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
# 가장 기본적인 코드가 됨.
# 숫자형 뽑아와서 8개 결측치 있는 거 확인하고 뚫려있는 애들만 true 나오도록.
# quantitative: 수치형
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
# 필요한 패키지 불러오기

# 현재 작업 디렉토리 확인
print(os.getcwd())

# 파일이 있는 디렉토리로 변경
os.chdir("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01")

# from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house/train.csv")
house_test=pd.read_csv("./data/house/test.csv")
sub_df=pd.read_csv("./data/house/sample_submission.csv")

# NaN 채우기
house_train.isna().sum()
house_test.isna().sum()

# 각 숫자변수는 평균채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

house_train[quant_selected].isna().sum()

# 각 범주형 변수는 최빈값으로 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna(house_train[col].mode()[0], inplace=True)

house_train[qual_selected].isna().sum()

train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include = [object]).columns,
    drop_first=True
    )
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

X = train_df.drop(["SalePrice", "Id"], axis = 1)
y = train_df['SalePrice']

# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)
# n_jobs = -1: 여러개의 코어에 각각의 배치를 해서 valid, train 부여함. -> 성능 향상됨. 속도 빨라짐

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(142, 146, 0.01)

mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df_result = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df_result

# 결과 시각화
plt.plot(df_result['lambda'], df_result['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()
# 최적의 alpha 값 찾기: 143.9
optimal_alpha = df_result['lambda'][np.argmin(df_result['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 라쏘 회귀 모델 생성
model = Lasso(alpha = 143.89)

# 모델 학습
model.fit(X, y) 

test_df=df.iloc[train_n:,]
test_df = test_df.drop(["SalePrice", "Id"], axis = 1)

quantitative = test_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    test_df[col].fillna(test_df[col].mean(), inplace=True)

test_df[quant_selected].isna().sum()

pred_y = model.predict(test_df)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/house/sample_submission12.csv", index=False)
