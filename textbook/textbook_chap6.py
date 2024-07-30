import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv("data/exam.csv")
# 조건에 맞는 행을 걸러내는 .query()
#두가지 방식이지만 query가 기억하기 더 쉽다.
# exam.query("nclass == 1")
exam.query("nclass == 1")
exam.query("nclass =/= 1")
exam.query("nclass == 1 & math > 50")
exam.query("nclass == 1 and math > 50")
exam.query("nclass == 1 | nclass == 2")
exam.query("nclass == 1 or nclass == 2")
exam.query("nclass in [1,2]")
exam.query("nclass not in [1,2]")
#exam.query[~exam["nclass"].isin([1,2])]


# 수학 점수가 50점 초과한 경우
exam.query('math > 50')
exam.query('math < 50')

#영어 점수
exam.query('english >= 50')
exam.query('english <= 80')

#1반이면서 수학 점수 50점 이상
exam.query('nclass == 1 & math >= 50')

#2반이면서 영어 점수 80점 이상
exam.query('nclass == 2 & english >= 80')

# 수학점수가 90점 이상이거나 영어 점수가 90점 이상인 경우
exam.query('math >= 90 | english >= 90')

# 영어 점수 90점 미만이거나 과학 점수 50점 미만
exam.query('englsih < 90 | science < 50')

# 1,3,5반에 해당하면 추출
exam.query('nclass == 1 | nclass == 3 | nclass == 5')
#다른 방법
exam.query('nclass in [1, 3, 5]')

#[]은 pandas series 형태로 나옴. 형태 유지하고 싶으면 대괄호 [] 하나 더 
exam["nclass"] > 3
exam[["nclass"]] > 3
exam[["id", "nclass"]]

#변수 제거
exam.drop(columns = ["math", "english"])
exam

exam.query("nclass == 1")[["math", "english"]]
exam.query("nclass == 1") \
          [["math", "english"]] \
          .head()
#정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = [True, False])

#변수 추가
exam = exam.assign(
                    total = exam["math"] + exam["english"]+ exam["science"], 
                    mean = (exam["math"] + exam["english"]+ exam["science"]) / 3
                    ).sort_values("total", ascending = False)
exam.head()
#lambda 함수 사용하기
#lambda는 total 사용가능 -> 짧아질 수 있다..
exam2 = pd.read_csv("data/exam.csv")
exam2 = exam2.assign(
                    total = lambda x: x["math"] + x["english"]+ x["science"], 
                    mean =  lambda x: x["total"] / 3
                    ) \
                    .sort_values("total", ascending = False)
exam2.head()

# 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \
     .agg(mean_math = ("math", "mean"))
     
# 과목별 평균
exam2.groupby('nclass') \
     .agg(mean_math = ('math', 'mean'),
         mean_english = ('english', 'mean'),
         mean_sc = ('science', 'mean'),
         )
import pydataset
import pandas as pd
mpg = pd.read_csv("data/mpg.csv")
mpg.query('category == "suv"') \
          .assign(total = (mpg['hwy'] + mpg['cty']) / 2)\
          .groupby('manufacturer') \
          .agg(mean_tot = ('total', 'mean')) \
          .sort_values('mean_tot', ascending = False)\
          .head()
mpg.head()

#0717 수업
# 중간고사 데이터 만들기
import pandas as pd

test1 = pd.DataFrame({'id'     : [1, 2, 3, 4, 5],
                      'midterm': [60, 80, 70, 90, 85]})
#기말고사 데이터
test2 = pd.DataFrame({'id'     : [1, 2, 3, 40, 5],
                      'final'  : [70, 83, 65, 95, 80]})
test1
test2
# Left join 왼쪽 기준으로
# left join이 가장 중요하다..
total = pd. merge(test1, test2, how="left", on="id")
total
# Right join 오른쪽 기준으로
total = pd. merge(test1, test2, how="right", on="id")
total
# Inner Join: 공통으로 가지고 있는 교집합 느낌.
total = pd. merge(test1, test2, how="inner", on="id")
total
# Outer join: 합집합
total = pd. merge(test1, test2, how="outer", on="id")
total
exam = pd.read_csv("data/exam.csv")
name = pd.DataFrame({'nclass'   : [1, 2, 3, 4, 5],
                     'teacher'  : ['kim', 'lee', 'park', 'choi', 'jung']})
name
pd.merge(exam, name, how="left", on="nclass")

# 데이터를 세로로 쌓는 방법
score1 = pd.DataFrame({'id'     : [1, 2, 3, 4, 5],
                      'score': [60, 80, 70, 90, 85]})
score2 = pd.DataFrame({'id'     : [6, 7, 8, 9, 10],
                      'score'  : [70, 83, 65, 95, 80]})
score1
score2
score_all = pd.concat([score1, score2])
score_all

test1
test2

pd.concat([test1, test2], axis=1)
