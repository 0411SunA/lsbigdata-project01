#교재 8장, p. 212
economics = pd.read_csv('data/economics.csv')
economics.head()

economics.info()
# object: 범주형 혹은 문자형/ 

# 선 그래프 시계열 그래프 만들기
import seaborn as sns
sns.lineplot( data=economics, x = "date", y = "unemploy")
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics["date"])
economics
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.day
# month_name() 은 괄호 쳐져있어서 메서드임. 
# 괄호 없으면 어트리뷰트
economics["date2"].dt.month_name()
economics["date2"].dt.quarter
economics["quarter"] = economics["date2"].dt.quarter
economics[["date2", "quarter"]]

# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()
# 30일 더하기 (month와 days와 차이 큼)
economics["date2"] + pd.DateOffset(days=30)
# 한 달 더하기 
economics["date2"] + pd.DateOffset(month=1)

# 연도 변수 만들기
economics['year'] = economics['date2'].dt.year
economics.head()

# x축에 연도 표시하기

# 신뢰구간 제거 (errorbar=None으로 설정하기)
sns.lineplot(data = economics, x = 'year', y = 'unemploy',
             errorbar = None)
sns.scatterplot(data = economics, x = 'year', y = 'unemploy')
plt.show()             
plt.clf()
economics.head()

# 신뢰구간을 각 년도에서 구할 수 있음.
# 각 년도의 표본평균과 표준편차를 구해라
# as_index=False 새로운 열로 만들기
my_df = economics.groupby('year', as_index=False) \
                 .agg(mon_mean = ("unemploy","mean"),
                  mon_std = ("unemploy", "std"),
                  mon_n = ("unemploy", "count")
                 )

my_df
mean + 1.96**std/sqrt(12)
my_df["left_ci"] = my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt

x = my_df["year"]
y = my_df["mon_mean"]

#plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.scatter(x, my_df["left_ci"], color="blue", s = 1)
plt.scatter(x, my_df["right_ci"], color="blue", s = 1)
plt.show()
plt.clf()
# 파란색 점 신뢰구간임.














