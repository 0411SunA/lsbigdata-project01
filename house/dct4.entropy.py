import numpy as np

#빨2 / 파3
p_r = 2/5
p_b = 3/5
h_zero = - p_r * np.log2(p_r) - p_b * np.log2(p_b)
h_zero
round(h_zero, 4) #0.971

#빨1 / 파3
p_r = 1/4
p_b = 3/4
h_1_r = - p_r * np.log2(p_r) - p_b * np.log2(p_b)
h_1_r
round(h_1_r, 4)

h_1_l = 0

h_1 = (1/5 * h_1_l) + (4/5 *h_1_r) # 가중치
round(h_1, 4) # 0.649

# Information Gain 
# 루트노드와 리프노드의 무질서도의 차이
# 얼마나 깔끔하게 정리했는지 정도
# 값이 클수록 난장판이 많이 수정됨

# 0.3이면 청소기만 돌린 정도 1이면 청소기 돌리고 걸레질하고 정리하고 쓰레기 내다버림
IG = h_zero - h_1
IG

#--------------------------------
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins = penguins.dropna()

df_X=penguins.drop("species", axis=1)
df_X = df_X[["bill_length_mm", "bill_depth_mm"]]
y = penguins[["species"]]

# 모델 생성
from sklearn.tree import DecisionTreeClassifier

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42)

param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(df_X,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_X,y)

from sklearn import tree
tree.plot_tree(model)


