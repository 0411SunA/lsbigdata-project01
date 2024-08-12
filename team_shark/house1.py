# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/train.csv")
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/test.csv")
sub_df = pd.read_csv('./data/house/sample_submission.csv')
