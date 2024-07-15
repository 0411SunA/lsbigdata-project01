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
