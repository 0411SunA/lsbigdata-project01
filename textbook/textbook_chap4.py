import numpy as np
import pandas as pd

df = pd.DataFrame({'name' :["김지훈", "이유진", "박동현", "김민지"],
              'english': [90, 80, 60, 70],
              'math': [50, 60, 100, 20]})
df
#series 형식
df["name"]
#dataframe형식
df[["name"]]

type(df)
type(df["name"])

sum(df["english"])/4

fruit = pd.DataFrame({'제품': ["사과", "딸기", "수박"],
                      '가격': [1800, 1500, 3000],
                      '판매량': [24, 38, 13]})
fruit                    
sum(fruit['가격']) / 3
sum(fruit['판매량']) / 3

df[["name", "english"]]
df["name"]

import pandas as pd
df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam['math'])    / 20
sum(df_exam['english']) / 20
sum(df_exam['science']) / 20

len(df_exam)
df_exam.shape
df_exam.size
#size = 20*5 = 100

#sheet 2 데이터 가져오고 싶을 때
df_exam = pd.read_excel("data/excel_exam.xlsx")
                        #,sheet_name = "Sheet2"
df_exam

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["mean"] = df_exam["total"]/3
df_exam

df_exam[df_exam["math"] >= 50]
df_exam[(df_exam["math"] >= 50) & (df_exam["english"] >= 50)]
(df_exam["math"] >= 50) & (df_exam["english"] >= 50)

#수학 평균보다못하는 사람, 영어도
#df_exam[(df_exam["math"] < df_exam["mean"])]
#df_exam[(df_exam["english"] < df_exam["mean"])]
#각 과목 평균
mean_m = df_exam["math"].mean()
mean_e = df_exam["english"].mean()
df_exam[(df_exam["math"] > mean_m) &
         df_exam["english"] < mean_e]
#이 중에서 3반인 학생들
df_nc3 = df_exam[df_exam["nclass"] == 3]
df_nc3[["math", "english", "science"]]
#math, english, science만 뽑아내기
df_nc3[0:1]
df_nc3[1:2]
df_nc3[1:5]
df_exam[8:16:2]
df_exam.sort_values("math", ascending=False)
df_exam.sort_values(["nclass", "math"], ascending =[True,False])

import numpy as np
a = np.array([4,2,5,3,6])
a[2]

#조건 만족하는 행 순서를 튜플로 알려줌
np.where(a > 3)
#numpy array 형태로 알려줌. dtype='<U4'
np.where(a > 3,"Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam
