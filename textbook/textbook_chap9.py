# ! pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
 
# 데이터 불러오기 
raw_welfare = pd.read_spss("./data/Koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

# 복사본 만들기
welfare = raw_welfare.copy()

# 데이터 검토
welfare
welfare.shape
#welfare.info()
#welfare.describe()

welfare=welfare.rename(
    columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10" : "marriage_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)
welfare = welfare[["sex", "birth", "marriage_type",
                    "religion", "income", "code_job", "code_region"]]
welfare.shape

welfare["sex"].dtypes

#결측치 확인
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1, 
                           "male", "female")
welfare["sex"].value_counts()

welfare["income"].describe()
#안채워진 데이터가 9884개임.
welfare["income"].isna().sum()

sum(welfare["income"] > 9998)

sex_income = welfare.dropna(subset="income") \
            .groupby("sex", as_index=False) \
            .agg(mean_income = ("income", "mean"))
sex_income

import seaborn as sns

sns.barplot(data = sex_income, x="sex", y = "mean_income",
            hue = "sex")
plt.show()
plt.clf()

# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 위 아래 검정색 막대기로  표시

welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x="age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income") \
                    .groupby("age", as_index=False) \
                    .agg(mean_income = ("income", "mean"))
sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기! <무응답자 수 그래프>
welfare["income"].isna().sum()

welfare["income"].isna()
my_df=welfare.assign(income_na=welfare["income"].isna()) \
                        .groupby("age", as_index=False) \
                        .agg(n = ("income_na", "sum"))


sns.barplot(data = my_df, x="age", y="n")
plt.show()
plt.clf()
# sum으로 해야 무응답자 수 알기위함임. 뒤쪽으로 갈수록 직업이 없어서 무응답 했을 것이다..
# 실제로 무응답자가 많았음. 2030은 무응답자 수가 적었음. 

#연령대별 월급 차이
                                
# 빈도 구하기
welfare['ageg'].value_counts
# 빈도 막대 그래프 만들기
sns.countplot(data = welfare, x = "ageg")
plt.show()
plt.clf()

#연령대별 월급 평균표 만들기
ageg_income = welfare.dropna(subset = "income")\
                     .groupby("ageg", as_index = False) \
                     .agg(mean_income = ("income", "mean")
                     )

# 그래프 만들기
sns.barplot(data = ageg_income, x = "ageg", y = "mean_income")

#막대 정렬하기
sns.barplot(data = ageg_income, x = "ageg", y = "mean_income",
            order = ["young", "middle", "old"])
# 0에서부터 9 10~19, 10대 20대 60대까지 해보아라
#연령대별 월급 차이
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 나이대별 수입 분석
# cut
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare=welfare.assign(
                age_group = pd.cut(welfare["age"], 
                bins=bin_cut, 
                labels=(np.arange(12) * 10).astype(str) + "대"))

# np.version.version
# (np.arange(12) * 10).astype(str) + "대"

age_income=welfare.dropna(subset="income") \
                    .groupby("age_group", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

age_income
sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

# 세 구간에 맞추어 나누고 싶음.
vec_x = np.random.randint(0, 100, 50)
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
# 나중에 생각해보기
# np.arange(12)*10 - 1
pd.cut(vec_x, bins=bin_cut) 
# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby+agg 콤보
# 안먹힘. 그래서 object 타입으로 바꿔 준 후 수행
welfare["age_group"]=welfare["age_group"].astype("object")
#-------------------------------------------------------------------
def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
    
sex_age_income
# -----------------------------------------------------------------
# 상위 4% 소득 순위
sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: np.quantile(x, q=0.96)))
    
sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="top4per_income", 
            hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!

import pandas as pd

# 예제 데이터 프레임 생성
# data = {
#     'income': [50000, 60000, 70000, None, 80000, 90000, None, 100000, 110000],
#     'age_group': ['20대', '30대', '40대', '20대', '30대', '40대', None, '20대', '30대'],
#     'sex': ['M', 'F', 'M', 'F', None, 'M', 'F', 'M', 'F']
# }
# welfare = pd.DataFrame(data)

# 9-6장
welfare["code_job"]
welfare["code_job"].value_counts()

## 직종 데이터 불러오기
list_job = pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx",
                         sheet_name = "직종코드")
list_job.head()                         

welfare = welfare.merge(list_job,
                        how='left', on='code_job')
welfare.dropna(subset=["job", "income"])[["income", "job"]]

#직업별 월급 평균표 만들기 # 쿼리 중간에 써도 됨.
job_income = welfare.dropna(subset = ["job", "income"])\
                    .query("sex=='female'")\
                    .groupby("job", as_index=False)\
                    .agg(mean_income = ("income", "mean"))
job_income.head()                    
#그래프 만들기 월급이 많은 직업

# 상위 10위 추출
top10 = job_income.sort_values("mean_income", ascending = False).head(10)
top10

# 맑은 고딕 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

# 막대 그래프 만들기
# 색깔 다르게 hue= "job"
sns.barplot(data = top10, y = "job", x = "mean_income", hue= "job")
plt.show()
plt.clf()

# 종교 유무에 따른 이혼율 분석하기
welfare.info()
welfare["marriage_type"]
df = welfare.query("marriage_type != 5") \
                    .groupby("religion", as_index=False)\
                    ["marriage_type"]\
                    .value_counts(normalize=True) # 비율 구하기가 핵심!
df = df.query("marriage_type ==1") \
       .assign(proportion=df["proportion"]*100)\
       .round(1)
# 노멀라이즈를 True로 하면 카운트 대신 비율로 나온다.








