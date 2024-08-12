import json

geo_seoul = json.load(open("./data/bigfile/SIG_Seoul.geojson", encoding = "UTF-8"))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])

# 행정 구역 코드 출력 
# 숫자가 바뀌면 "구"가 바뀌는 구나!
geo_seoul["name"]
geo_seoul["features"][0].keys()
geo_seoul["features"][0]["properties"]
geo_seoul["features"][1]["properties"]
geo_seoul["features"][2]["properties"]
geo_seoul["features"][-1]["properties"]

# 위도, 경도 좌표 출력
geo_seoul["features"][0]["geometry"]
# 리스트
coordinate_list = geo_seoul["features"][1]["geometry"]["coordinates"]
# 리스트가 결국 1개여서 우린 25장 중 1장을 꺼냈기 때문에 
len(coordinate_list)

type(geo_seoul["features"][0]["geometry"]["coordinates"])

# 리스트 대괄호 없애는 과정
#우리가 진짜 보고싶어하는 위도, 경도 정보를 알기 위해 [0][0]을 처리함.
len(coordinate_list[0][0])
coordinate_list[0][0]

import numpy as np
import matplotlib.pyplot as plt

# x, y를 뽑기 위해서 array로 바꿈 일자로 변경! 어레이로 바꾸면 loc 없이
# 바로 리스트 해서 슬라이싱 가능함.
coordinate_array = np.array(coordinate_list[0][0])
x = coordinate_array[:, 0]
y = coordinate_array[:, 1]
len(x)

plt.plot(x[::10], y[::10])
plt.show()
plt.clf()

# 함수로 만들기

def draw_seoul(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    # 축 비율 1:1로 설정
    plt.axis('equal')
    plt.show()
    plt.clf()
    
    return None
# 함수에 리턴 없어도 시행 됨. 따라서 논 쓴거임.
draw_seoul(12)

# 서울시 전체 지도 그리기
# gu_name | x | y
# ================
# 종로구  |126|36
# 종로구  |126|36
# 종로구  |126|36
# .........
# 종로구  |126|36
# 중구    |125|38
# 중구    |125|38
# 중구    |125|38
# .........

gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
gu_name

coordinate_list = [geo_seoul["features"][x]["geometry"]['coordinates'] for x in range(len(geo_seoul["features"]))]
coordinate_list

np.array(coordinate_list[0][0][0])

# np.array(coordinate_list[0][0][0]) # 종로
# np.array(coordinate_list[1][0][0]) # 중구
# np.array(coordinate_list[2][0][0]) # 용산구
# np.array(coordinate_list[3][0][0]) # 성동구

import numpy as np
import pandas as pd

# 남규님 code - 종로구
pd.DataFrame({'gu_name' : [gu_name[0]] * len(np.array(coordinate_list[0][0][0])),
              'x'       : np.array(coordinate_list[0][0][0])[:,0],
              'y'       : np.array(coordinate_list[0][0][0])[:,1]})
              
# 한결 생각 - 얘는 왜인지 모르겠으나 syntax error 발생
# pd.DataFrame({'gu_name' : [gu_name[x]] * len(np.array(coordinate_list[x][0][0]) for x in range(len(geo_seoul["features"]))],
#               'x'       : [np.array(coordinate_list[x][0][0])[:,0] for x in range(len(geo_seoul["features"]))],
#               'y'       : [np.array(coordinate_list[x][0][0])[:,1]] for x in range(len(geo_seoul["features"])) })

# 빈 리스트 생성
empty = []

# for in 구문을 이용하하여 geo_seoul["features"]의 길이만큼 for 문 안의 내용을 반복
for x in range(len(geo_seoul["features"])):
    df = pd.DataFrame({
        'gu_name': [gu_name[x]] * len(np.array(coordinate_list[x][0][0])),
        'x': np.array(coordinate_list[x][0][0])[:, 0],
        'y': np.array(coordinate_list[x][0][0])[:, 1]
    })
    empty.append(df)

# 모든 DataFrame을 하나로 합치기, ignore_index=True를 이용하여 기존의 인덱스를 무시하고 새로운 인덱스 부여
seoul_total = pd.concat(empty, ignore_index=True)
seoul_total

# 그룹화 ㄱㄱ?
import seaborn as sns
sns.scatterplot(data = seoul_total, x='x', y='y', hue="gu_name", s=1)
# plt.plot(x,y, hue="gu_name")
plt.show()
plt.clf()

# -------수업-----
# 구 이름 만들기
gu_name = geo_seoul["features"][0]["properties"]["SIG_KOR_NM"]
gu_name
gu_name = list()
for i in range(25):
    #gu_name = gu_name + geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
gu_name

# 방법1
gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
gu_name
#방법2
# gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(25))]
# gu_name
# 함수
def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
    return pd.DataFrame({"gu_name": gu_name, "x": x, "y": y})

make_seouldf(1)

#---------------

plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
#----------------------
#-----오후 수업
# x, y 판다스 데이터 프레임
import pandas as pd

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)    

result

# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data=result,
    x='x', y='y', hue='gu_name', s=2)
plt.show()
plt.clf()


# 서울 그래프 그리기
import seaborn as sns
gangnam_df=result.assign(is_gangnam=np.where(result["gu_name"]!="강남구", "안강남", "강남"))
sns.scatterplot(
    data=gangnam_df,
    x='x', y='y', legend=False, 
    palette={"안강남": "grey", "강남": "red"},
    hue='is_gangnam', s=2)
plt.show()
plt.clf()

#--------교재
import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("./data/bigfile/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
# code를 그래프에 나타낼 수 있음. 숫자형 말고 object형이어야 그래프 가능
df_seoulpop.info()

# 패키지 설치하기
# !pip install folium
import folium

center_x = result["x"].mean()
center_y = result["y"].mean()
# p. 304
# 흰 도화지 맵 구하기
map_sig = folium.Map(location = [37.551, 126.973],
           zoom_start = 12,
           tiles="cartodbpositron")
map_sig.save("map_seoul.html")

# 코로플릿 (기본 틀에 색 별로 지역 표시하기)
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    fill_color = "viridis",
    bins = bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)


# 각 구별 인구 수 
bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
# 경계선 값 나오게 하기 하위 20퍼센트가 갖고 있는 인구수 : 296543.6
bins

# 점 찍는 법 (marker 위치 x,y 다르게 설정하기)
# make_seouldf(0) 종로구 뽑아올 수 있는 거
make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")    

# folium marker로 집 위치 다 찍기













