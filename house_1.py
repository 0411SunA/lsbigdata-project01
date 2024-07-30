import numpy as np
import pandas as pd

house_df = pd.read_csv('./data/house/train.csv')
house_df.shape
house_df.head()
house_df.info()

price_mean = house_df["SalePrice"].mean()
price_mean

sub_df = pd.read_csv('./data/house/sample_submission.csv')
sub_df

# house_df의 price_mean으로 sub_df['SalePrice']로 대체해라
sub_df['SalePrice'] = price_mean
sub_df

# 지금 있는 sub_df를 같은 이름의 파일로 내보내라 바꿔치기됨.
sub_df.to_csv("./data/house/sample_submission.csv", index=False)


# 0729 문제
# train 에 있는 정보들에서 집값 정보 중에 yearbuilt하면 언제 지어졌는지 알 수 있음. 같은 해에 지어진 집을
# 하나로 보기 연도마다 평균이 있을 것. test set에 있는 집값을 예측해보아라. 

house_df = pd.read_csv('./data/house/train.csv')
house_df.shape
house_df.head()
house_df.info()

my_df = house_df.groupby('YearBuilt', as_index=False) \
                 .agg(mean = ("SalePrice","mean"),)
my_df

test_df = pd.read_csv('./data/house/test.csv')
test_df

# house_df의 price_mean으로 sub_df['SalePrice']로 대체해라
merged_df = pd.merge(test_df, my_df, how = 'left', on='YearBuilt')
sub_df['SalePrice'] = merged_df['mean']
sub_df['SalePrice'] = sub_df['SalePrice'].fillna(price_mean)

sub_df.to_csv("./data/house/sample_submission.csv", index=False)



