import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
ship_tr = pd.read_csv("data/titanic/train.csv")
ship_test = pd.read_csv("data/titanic/test.csv")
ship_df = pd.read_csv("data/titanic/sample_submission.csv")

# 전처리: 결측값 처리
# 수치형 데이터 결측값 평균으로 채우기
quantitative = ship_tr.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    ship_tr[col].fillna(ship_tr[col].mean(), inplace=True)

# 범주형 데이터 결측값 최빈값으로 채우기
qualitative = ship_tr.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    ship_tr[col].fillna(ship_tr[col].mode()[0], inplace=True)

# 테스트 데이터에 대해서도 동일한 전처리 수행
quantitative = ship_test.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    ship_test[col].fillna(ship_test[col].mean(), inplace=True)

qualitative = ship_test.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    ship_test[col].fillna(ship_test[col].mode()[0], inplace=True)

# 데이터 통합 및 더미 코딩
train_n = len(ship_tr)
df = pd.concat([ship_tr, ship_test], ignore_index=True)
df = df.drop(["PassengerId", "Name"], axis=1)

# 범주형 데이터 더미 코딩
col = df.select_dtypes(include=[object]).columns
col = col[:-1]  # 마지막 열 제외
df = pd.get_dummies(df, columns=col, drop_first=True)

# 학습 및 테스트 데이터 분할
train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

train_x = train_df.drop("Transported", axis=1)
train_y = train_df["Transported"].astype("bool")  # 타겟 변수가 "Transported"로 설정된 경우
test_x = test_df.drop("Transported", axis=1)

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 첫 번째 층 모델 학습 (RandomForest, XGBoost, LightGBM)
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [200],
    'max_depth': [7],
    'min_samples_split': [20],
    'min_samples_leaf': [5]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring='accuracy', cv=5)
grid_search_rf.fit(train_x_scaled, train_y)
best_rf_model = grid_search_rf.best_estimator_

xgb_model = XGBClassifier(random_state=42)
param_grid_xgb = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [7]
}
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=5)
grid_search_xgb.fit(train_x_scaled, train_y)
best_xgb_model = grid_search_xgb.best_estimator_

lgbm_model = LGBMClassifier(random_state=42)
param_grid_lgbm = {
    'learning_rate': [0.05],
    'n_estimators': [200],
    'max_depth': [7]
}
grid_search_lgbm = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, scoring='accuracy', cv=5)
grid_search_lgbm.fit(train_x_scaled, train_y)
best_lgbm_model = grid_search_lgbm.best_estimator_

# 첫 번째 층 모델 예측값 생성
y1_hat = best_rf_model.predict(train_x_scaled)
y2_hat = best_xgb_model.predict(train_x_scaled)
y3_hat = best_lgbm_model.predict(train_x_scaled)

train_x_stack_1 = pd.DataFrame({'y1': y1_hat, 'y2': y2_hat, 'y3': y3_hat})

# 두 번째 층 모델 학습 (GradientBoostingClassifier + RandomForestClassifier)
gb_model = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'learning_rate': [0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5]
}
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='accuracy', cv=5)
grid_search_gb.fit(train_x_stack_1, train_y)
best_gb_model = grid_search_gb.best_estimator_

# 두 번째 층에 RandomForest 추가
rf_model_2 = RandomForestClassifier(random_state=42)
param_grid_rf_2 = {
    'n_estimators': [200],
    'max_depth': [5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}
grid_search_rf_2 = GridSearchCV(estimator=rf_model_2, param_grid=param_grid_rf_2, scoring='accuracy', cv=5)
grid_search_rf_2.fit(train_x_stack_1, train_y)
best_rf_model_2 = grid_search_rf_2.best_estimator_

# 새로운 모델 CatBoostClassifier 추가
cat_model = CatBoostClassifier(verbose=0, random_state=42)
param_grid_cat = {
   
