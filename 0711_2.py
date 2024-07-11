#교재 63페이지
#seaborn 패키지 설치
#! pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var

seaborn.countplot( x = var)
plt.show()
plt.clf()

# hue는 색을 결정함.
df = sns.load_dataset("titanic")
sns.countplot(data = df, x = "sex", hue="sex")
plt.show()
plt.clf()

?sns.countplot
sns.countplot(data=df, x = "class")
sns.countplot(data = df, x = "class", hue = "alive")
sns.countplot(data = df, 
              y = "class",
              hue = "alive",
             )
plt.show()
# orient   orient="v" 또는 "h" 사용해서 데이터만 있을 때 지정 가능함.

!pip install scikit-sklearn
import sklearn.metrics

sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

import sklearn.metrics as met
met_accuracy_score()
