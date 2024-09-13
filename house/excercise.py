#  워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
import pandas as pd
import numpy as np

# 파일이 있는 디렉토리로 변경
os.chdir("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project01")

df = pd.read_csv("./data/leukemia_remission.txt", sep='\t')
print(df.shape)

df.head()

# 문제 1. 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP",
                         data=df).fit()
print(model.summary())
# 문제 2. 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
# 답: 통계적으로 유의하다. 이유: LLR p-value가 0.0467인데 유의수준 0.05보다 작으므로 유의하다 

# 문제 3. 유의수준 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# 답: 2개 유의수준 0.2보다 작은 변수(LI, TEMP) 

# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?
# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9
my_odds = np.exp(64.2581 +30.8301*0.65 + 24.686316*0.45 -24.9745*0.55 +4.3605*1.2 -0.0115*1.1 -100.1734*0.9)
my_odds

# 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
leuk = my_odds / (my_odds+1)
leuk

# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 
# TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
# TEMP 계수: -100.1734
# TEMP가 1단위 증가하면 백핼병 세포가 관측되지 않을 확률의
# 오즈가 e^(-100.1734)배 감소한다는 것을 의미한다.
# 즉 체온이 높아질수록 백혈병 세포가 관측될 확률이 높아진다. 

# 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
cell_coef = 30.8301 
cell_std_err = 52.135 
z = norm.ppf(0.995,0,1)
upper = cell_coef + z * cell_std_err
lower = cell_coef - z * cell_std_err

# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 
# 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
# 예측 확률 계산
data['df'] = model.predict(X)
data['Predicted_Classes'] = (data['Predicted_Probabilities'] >= 0.5).astype(int)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(data['REMISS'], data['Predicted_Classes'])
print("\n혼동 행렬:")
print(conf_matrix)

#혼동 행렬:
#[[15  3]
#[ 4  5]]
# 문제 9. 해당 모델의 Accuracy는 얼마인가요?
(15+5)/(15+3+4+5)
# 문제 10. 해당 모델의 F1 Score를 구하세요.

precision = 15/(15+4)
recall = 15/(15+3)
F1_score = 2* (precision*recall/(precision + recall))