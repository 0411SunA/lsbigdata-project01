# !pip install plotly

# plotly.graph_objects: 그래프의 정석, 완벽히 커스터마이징 가능하지만 복잡함
import plotly.graph_objects as go
import plotly.express as px
# plotly 빠른데 컨트롤 힘들다

# 현재 워킹디렉토리 설정하는 법
# import os
# current working directory
# cwd = os.getcwd()
# cwd
# os.chdir('c:\\Users\\USER\\Documents\\LS빅데이터스쿨\\lsbigdata-project01\\')

# 주석처리: Ctrl + /

import pandas as pd
import numpy as np

df_covid19_100=pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/df_covid19_100.csv")
df_covid19_100.info()

fig = go.Figure(
    data = {"type": "scatter",
         "mode": "markers",
         "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
         "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
         "marker": {"color": "red"}
         }
)
fig.show()

# p.26 마진 변수 설정 레이아웃 top botttom 등
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
fig = go.Figure(
    data = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout= {
        "title": "코로나 19 발생현황",
        "xaxis": {"title": "날짜", "showgrid": False},
        "yaxis": {"title": "확진자수"},
        "margin": margins_P
    }
)
fig.show()

# 프레임속성을 이용한 애니메이션

# 애니메이션 프레임 생성
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date) # 무조건 str 인지 확인. 그래프 그리기 위해서 숫자면 안됨
    }
    frames.append(frame_data)
    

# x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]

# 애니메이션을 위한 레이아웃 설정 다꾸하는 느낌
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()


