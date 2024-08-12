import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins
import plotly.graph_objects as go

penguins = load_penguins()

# 점 크기를 고정된 값으로 설정합니다.
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    #trendline="ols" # p.134 심슨스 패러독스 변수 고려한 흐름 나타내기
)
fig.show()
# 점 크기와 투명도를 설정합니다.
fig.update_traces(
    marker=dict(size=12, opacity=0.6)  # 점의 크기를 12로, 투명도를 0.6으로 설정합니다.
)

# 레이아웃 업데이트 dict 딕셔너리 대신 쓰는거 (key, value)
fig.update_layout(
    title=dict(
        text="팔머펭귄 종별 부리 길이 vs. 깊이",
        font=dict(color="white", size=24)  # 제목의 크기를 조정합니다.
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        font=dict(color="white"),
        title=dict(text="펭귄 종", font=dict(color="white"))  # 범례 제목을 변경합니다.
    ),
)

fig.show()
#--------------------
from sklearn.linear_model import LinearRegression 

model = LinearRegression()

# 결측치 있어서 빼주기
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit=model.predict(x)

fig.add_trace( # trace: 이미 그려진 그래프에 더해라
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()
# 잠복변수 lurk 심슨스 패러독스 때문에 회귀분석 할 때 항상 조심한다. 잠재 변수 때문에
# 흐름이 바뀔 수 있음

model.coef_      # 기울기 a 부리길이가 1mm 증가할 때마다 부리 깊이가 0.08만큼 줄어든다. 
model.intercept_ # 절편 b

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
#drop_first=True 이거 원래 false 넣었었는데 두 종만 알아도 아니까 true로 변경함
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]
# x와 y 설정
# 3개의 선택지가 있지만 2개만 있어도 어느 종인지 알 수 있음. TF면 친스트랩 이런식으로!
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_
#---------------------------
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                legend=False)
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,
                color="black")
plt.show()
plt.clf()
#----------------------------
# 회귀분석 만들기
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
0.2 * 40.5 -1.93 * True -5.1* False + 10.56

regline_y = model.predict(x)
import matplotlib.pyplot as plt
plt.scatter(x["bill_length_mm"], y, color = "black", s = 1,
hue = penguins["species"])
plt.scatter(x["bill_length_mm"], regline_y, s=1)
plt.show()
plt.clf()
