import pandas as pd
import numpy as np

#데이터 탐색 함수
#head()
#tail()
#shape
#info()
#describe()

exam = pd.read_csv("data/exam.csv")
exam.head(10)
exam.tail(10)
exam.shape
# 메서드 vs. 속성(어트리뷰트)
#shape은 어트리뷰트임. 어떻게 구분할까? 
#설명란에 속성은 괄호 없음. 메서드는 괄호 있음.
exam.info()
#info는 데이터에 대한 정보
#describe는 데이터에 대한 요약된 정보
exam.describe()
type(exam)
#pandas dataframe이 속성임.
var=[1,2,3]
type(var)
exam.head()
# var.head() -> 오류남. head 메서드가 없어서!

exam2 = exam.copy()
exam2 = exam2.rename(columns={'nclass' : 'class'})
exam2.head()
exam2['total'] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()

#200점 이상은 pass 미만은 fail
exam2["test"] = np.where(exam2['total'] >= 200, 'pass', 'fail')
exam2.head()

#200 이상: A
#100 이상: B
#100 미만: C
exam2["test2"] = np.where(exam2['total'] >= 200, 'A',
                 np.where(exam2['total'] >= 100, 'B','C'))
exam2.head()

#128 혼자해보기!
exam2["test2"].isin(["A", "C"])

import matplotlib.pyplot as plt
count_test=exam2['test'].value_counts()
count_test.plot.bar(rot = 0)
#rot 설정 외우진 마라.
plt.show()
plt.clf()
