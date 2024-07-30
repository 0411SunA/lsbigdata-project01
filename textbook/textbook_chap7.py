import pandas as pd
import numpy as np

df = pd.DataFrame({"sex"  : ['M', 'F', np.nan, 'M', 'F'],
                   "score": [5, 4, 3, 4, np.nan]})
df                   
df["score"] + 1

#np.nan인 것만 true표시됨
pd.isna(df).sum()

#결측치 제거하기
df.dropna(subset = "score")          # score 변수에서 결측치 제거
df.dropna(subset = ["score", "sex"]) # 여러 변수 결측치 제거법
df.dropna()                          # 모든 변수 결측치 제거

exam = pd.read_csv("data/exam.csv")

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
# 행 인덱스는 숫자 가능, 열 인덱스는 iloc 사용하기
exam.iloc[0:2, 0:4]
# exam.loc[[2,7,4], ["math"]] = np.nan
exam.iloc[[2,7,4], 2] = np.nan
exam.iloc[[2,7,4], 2] = 3
exam

#수학점수 50점 이하인 학생들 점수 50점으로 상향 조정!
exam.loc[exam["math"] <= 50, ["math"]] = 50
exam
# 영어 점수 90점 이상 90점으로 하향 조정 (iloc 사용)
# iloc을 사용해서 조회하려면 무조건 숫자벡터가 들어가야 함.
exam.iloc[exam["english"] >= 90,"english"]           #실행 안됨
exam.iloc[exam["english"] >= 90, 3] = 90
exam.iloc[exam[exam["english"] >= 90].index, 3] = 90 #실행 됨 index 벡터도 작동
exam.iloc[np.where(exam["english"] >= 90)[0], 3]     #np.where도 튜플이라 [0] 사용해서 꺼내오면 됨.
exam

# math 점수 50점 이하 "-"" 변경
exam.loc[exam["math"] <= 50, "math"] = "-"
exam

# "-" 결측치를 수학점수 평균으로 바꾸고 싶은 경우
#내 답안: exam.loc[exam["math"] == "-" , "math"] = exam["math"].mean()
#exam
# 방법1
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean
exam
# 방법2
math_mean =exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math']=="-", "math"] = math_mean()
exam
# 방법 3
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam['math']=='-','math'] = math_mean
exam
# 방법 4
exam.loc[exam["math"] =="-", ["math"]] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean
exam

# 방법 5 이건 아직 몰라도 됨..미리 예방주사 맞았다고 생각.. 5번 굳이? 싶다
vector = 
math_mean = np.nonmean([np.array if x == "-" else float(x) for x in exam["math"]])
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])
exam

#방법 6
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)
exam


df.loc[df["score"] == 3.0, ["score"]] = 4
df

exam.loc[[0], ]
exam.loc[[0], ["id", "nclass"]]
exam.loc[[2, 7, 14],]
