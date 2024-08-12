import numpy as np
import pandas as pd

house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/train.csv")
df = house_train.sort_values('SalePrice').head(10)

house_train = house_train[["Id", "BldgType", "Neighborhood", "RoofStyle", "SalePrice"]]

# 연도별 평균 
house_mean = house_train.groupby(["BldgType", "Neighborhood", "RoofStyle"], as_index = False) \
                        .agg(mean = ('SalePrice', 'mean'))
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01/data/house/test.csv")
house_test = house_test[["Id", "BldgType", "Neighborhood", "RoofStyle"]]

house_test = pd.merge(house_test, house_mean, how ='left', on = ["BldgType", "Neighborhood", "RoofStyle"])
house_test = house_test.rename(columns = {'mean' : 'SalePrice'})

house_test.isna().sum()  # na 값 세기 
house_test.loc[house_test['SalePrice'].isna()] # na값 보기 

 # na 값 채우기 
price_mean = house_train["SalePrice"].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(price_mean) 

#sub_df 불러오기
sub_df = pd.read_csv('./data/house/sample_submission.csv')
sub_df
#SalePrice 바꿔치기 및 저장 
sub_df['SalePrice'] = house_test['SalePrice']
sub_df.to_csv("./data/house/sample_submission.csv", index = False)
