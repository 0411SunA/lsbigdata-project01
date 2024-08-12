# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import folium

## 필요한 데이터 불러오기
# house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/train.csv")
# house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/test.csv")
# sub_df = pd.read_csv('./data/house/sample_submission.csv')
house_loc = pd.read_csv('./data/house/house_loc.csv')
house_loc
# Longitude: 경도   Latitude: 위도
house_loc = house_loc.iloc[:, -2:]
len(house_loc)

# 위도와 경도 평균 구하기
house_loc["Longitude"].mean()
house_loc["Latitude"].mean()

# 흰 도화지 맵 만들기
map_sig = folium.Map(location = [42.03448223395904, -93.64289689856655],
           zoom_start = 12,
           tiles="cartodbpositron")

# 중심점 찍기 (marker 위치 x,y 다르게 설정하기)
make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([42.03448223395904, -93.64289689856655], popup="Ames").add_to(map_sig)
len(house_loc)
# iloc[i, 1] i번째 행의 (0,1)열 중 1을 가져오기 위해 1로 설정(위도)
# iloc[i, 0] i번째 행의 (0,1)열 중 0을 가져오기 위해 0으로 설정(경도)
for i in range(2930):
    folium.Marker([house_loc.iloc[i,1], house_loc.iloc[i,0]], popup=str(i)).add_to(map_sig)

map_sig.save("map_ames.html")

# ames 전체 집 좌표 점 찍기
