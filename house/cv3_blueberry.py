import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
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

## 필요한 데이터 불러오기
berry_train=pd.read_csv("./data/blueberry/train.csv")
berry_test=pd.read_csv("./data/blueberry/test.csv")
sub_df=pd.read_csv("./data/blueberry/sample_submission.csv")

X = berry_train.drop(['yield', 'id', 'MinOfUpperTRange', 
                      'MaxOfUpperTRange', 'MaxOfLowerTRange', 
                      'MinOfLowerTRange', 'AverageOfLowerTRange', 'AverageRainingDays'], axis=1)
y = berry_train["yield"]
berry_test = berry_test.drop(['id', 'MinOfUpperTRange', 
                      'MaxOfUpperTRange', 'MaxOfLowerTRange', 
                      'MinOfLowerTRange', 'AverageOfLowerTRange', 'AverageRainingDays'], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(berry_test)

# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

# rmae 계산 함수
def rmae(model, X, y):
    score = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean())
    return score

# 1. Lasso 회귀에 대해 alpha 값을 여러 개 시도
lasso_alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.3, 3.0]
lasso_rmaes = []

for alpha in lasso_alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_rmae = rmae(lasso_model, X_scaled, y)
    lasso_rmaes.append(lasso_rmae)
    print(f"Lasso (alpha={alpha}) rmae (cross-validated): {lasso_rmae}")

# Lasso에서 가장 좋은 alpha 값 선택: 1.0
optimal_lasso_alpha = lasso_alphas[np.argmin(lasso_rmaes)]
print(f"Optimal alpha for Lasso: {optimal_lasso_alpha}")

# 2. Ridge 회귀에 대해 alpha 값을 여러 개 시도 
ridge_alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ridge_rmaes = []

for alpha in ridge_alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_rmae = rmae(ridge_model, X_scaled, y)
    ridge_rmaes.append(ridge_rmae)
    print(f"Ridge (alpha={alpha}) rmae (cross-validated): {ridge_rmae}")

# Ridge에서 가장 좋은 alpha 값 선택: 0.5 
optimal_ridge_alpha = ridge_alphas[np.argmin(ridge_rmaes)]
print(f"Optimal alpha for Ridge: {optimal_ridge_alpha}")

# 3. KNN 회귀에 대해 k 값 시도
k_values = [3, 5, 7, 9, 11, 13, 15]
knn_rmaes = []

for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_rmae = rmae(knn_model, X_scaled, y)
    knn_rmaes.append(knn_rmae)
    print(f"KNN (k={k}) rmae (cross-validated): {knn_rmae}")

# KNN에서 가장 좋은 k 값 선택: 13
optimal_k = k_values[np.argmin(knn_rmaes)]
print(f"Optimal k for KNN: {optimal_k}")

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor

# Bagging 모델 적용 (최적 alpha 및 k 값 사용)
bagging_lasso = BaggingRegressor(estimator=Lasso(alpha=optimal_lasso_alpha), n_estimators=10, random_state=2024)
bagging_ridge = BaggingRegressor(estimator=Ridge(alpha=optimal_ridge_alpha), n_estimators=10, random_state=2024)
bagging_knn = BaggingRegressor(estimator=KNeighborsRegressor(n_neighbors=optimal_k), n_estimators=10, random_state=2024)

# 각 Bagging 모델의 교차 검증 수행
bagging_lasso_rmae = rmae(bagging_lasso, X_scaled, y)
print(f"Bagging Lasso rmae (cross-validated): {bagging_lasso_rmae}")

bagging_ridge_rmae = rmae(bagging_ridge, X_scaled, y)
print(f"Bagging Ridge rmae (cross-validated): {bagging_ridge_rmae}")

bagging_knn_rmae = rmae(bagging_knn, X_scaled, y)
print(f"Bagging KNN rmae (cross-validated): {bagging_knn_rmae}")

# Bagging 모델 학습 및 예측
bagging_lasso.fit(X_scaled, y)
bagging_ridge.fit(X_scaled, y)
bagging_knn.fit(X_scaled, y)

# 각 Bagging 모델의 예측값 계산
pred_y_lasso = bagging_lasso.predict(test_X_scaled)
pred_y_ridge = bagging_ridge.predict(test_X_scaled)
pred_y_knn = bagging_knn.predict(test_X_scaled)

# 최종 예측값을 앙상블 (세 모델 예측 평균)
final_prediction = (pred_y_lasso + pred_y_ridge) / 2

# 최종 예측값을 제출 파일로 저장
sub_df["yield"] = final_prediction
sub_df.to_csv('final_submission.csv', index=False)
print("Final submission file saved!")
