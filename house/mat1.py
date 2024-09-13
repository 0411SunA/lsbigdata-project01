# 3가지 방법으로 회귀분석 계수 구하기
import numpy as np

# 벡터 * 벡터 (내적)
a = np.arange(1, 4)
a
b = np.array([3, 6, 9])
b
b.shape

a.dot(b)

# 행렬 * 벡터 (곱셈)
a = np.array([1, 2, 3, 4]).reshape((2, 2), order = 'F')
a

b = np.array([5, 6]).reshape(2, 1)
b

# a * b 임 @로 대체 가능
a.dot(b)
a @ b

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2)
                                   , order = 'F') # 세로로 쌓아줌order = 'F'
b = np.array([5, 6, 7, 8]).reshape((2, 2)
                                   , order = 'F')
a
b
a @ b
# Q1
a = np.array([1, 2, 1, 0, 2, 3]).reshape((2, 3)
                                   , order = 'F')
b = np.array([1, 0, -1, 1, 2, 3 ]).reshape((3, 2)
                                   , order = 'F')
a
b
a @ b

# Q2
np.eye(3)
a = np.array([3, 5, 7,
              2, 4, 9,
              3, 1, 0]).reshape(3, 3) 

a @ np.eye(3)
np.eye(3) @ a

# transpose
a
a.transpose # 대각선을 기준으로 뒤집어준다
b = a[:, 0:2]
b
b.transpose()

# 회귀분석 데이터행렬 부리길이 예측할 때 사용 (펭귄 2마리 예측해보기)
x=np.array([13, 15,
            12, 14,
            10, 11,
            5, 6]).reshape(4, 2)
    
vec1=np.repeat(1, 4).reshape(4, 1) # 베타 0에 곱해지는 상수 1
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1) # y는 정답
matX 

beta_vec=np.array([2, 0, 1]).reshape(3, 1)
beta_vec
matX @ beta_vec # 예측식

(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec) 
# 손실함수 (정답 - 예측 ) 전치행렬하고 제곱
#-------------------------------
# 역행렬 a inverse
a = np.array([1, 5, 3, 4]).reshape(2,2)
a_inv = (-1/11) * np.array([4. -5 -3, 1])
a @ a_inv

## 3 by 3 역행렬
a = np.array([-4, -6, 2,
               -5, -1, 3,
               -2, 4, -3]).reshape(3, 3)
a_inv = np.linalg.inv(a)
np.linalg.det(a)
a_inv

np.round(a @ a_inv, 3)
# 주의! 역행렬 항상 존재하는 것 아님
# 행렬의 세로 벡터들이 선형 독립일때만 역행렬을 구할 수 있음
# '선형독립'이 아닌 경우: 선형종속 (공통점 찾음)
# 역행렬이 존재하지 않는다 = 특이행렬 (singular matrix) = 행렬식이 0이다
b = np.array([1, 2, 3,
              4, 5, 6,
              7, 8, 9]).reshape(3, 3)
b_inv = np.linalg.inv(b) # 에러남 (정상) 
np.linalg.det(b) # 행렬식이 항상 0 나오면 선형종속이다. det() 선형종속인지 아닌지 판단해줌


# 벡터 형태로 베타 구하기
matX
y
# 베타 구하기 직사각형 넓이 최소가 되도록
XtX_inv=np.linalg.inv((matX.transpose() @ matX))
Xty=matX.transpose() @ y # y가 정답
beta_hat=XtX_inv @ Xty
beta_hat

# 모델 fit으로 베타 구하기
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(matX[:, 1:], y) # 1이 y절편 곱해주는 값이었기에 x값에는 필요없어서 1제외

model.intercept_
model.coef_
#-------------------------------------------------------------
# minimize로 Ridge beta 구하기(패널티 있는 손실함수)
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta) # a가 y_hat임, a가 오차임. 
    return (a.transpose() @ a) + 3*np.abs(beta).sum() 
#(람다 3으로 고정해줬으니까) 람다클수록 더 많은 계수들 없어짐
# 3*np.abs(beta).sum() 이 부분 더했을 뿐인데! 패널티 추가되었기 때문에 라쏘 됨. 

line_perform([8.55,  5.96, -4.38]) # 원래 회귀분석 계수 (패널티 없는) 지금은 필요 x
line_perform([3.76,  1.36, 0]) # 원래 회귀분석 계수 (패널티 없는) 지금은 필요 x

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기 (오차가 작은 게 좋으니까!)
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun) #결과, 부리깊이
print("최소값을 갖는 x 값:", result.x) # 결과 나오기 위한 x값
