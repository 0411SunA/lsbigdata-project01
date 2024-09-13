from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

Bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=50,
                                  max_samples=100,
                                  n_jobs=-1, random_state=42)

# ---내 필기-----------------------------
# 모델 50개, 한 데이터 세트는 100개, n_jobs는 프로세싱 한번에 1개 프로세스 수행
# random_state는 무작위로 숫자 선택할 때 사용
#----------------------------------------
# * n_estimator: Bagging에 사용될 모델 개수
# max_sample: 데이터셋 만들 때 뽑을 표본 크기

# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier(n_estimators=50,
                                max_leaf_nodes=16,
                                n_jobs=-1, random_state=42)

# rf_model.fit(X_train, y_train)