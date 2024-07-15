# 0711배운거 복습
a = 1, 2, 3
a = (1,2,3) 
a

a = [1,2,3]
a

#soft copy
b = a
b

# 두번째 원소에 4로 변경
a[1] = 4
a

b

b
id(a)
id(b)

#b도 똑같이 불러오라는 정보. a에 있는 정보를 b로 옮겨라 따라서 deepcopy해야함.

#deep copy 정보 저장은 다르게하는거.
a = [1,2,3]
a

id(a)

#1,2번째 방법
b=a[:]
b = a.copy()
id(b)

a[1] = 4
a
b

# 수학함수
import math

x = 4
math.sqrt(x)

#지수 계산 예제
exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

#로그 계산 예제
log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

#팩토리얼 계산 예제
fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

# 확률밀도함수 값 계산
import math
def my_normal_pdf(x, mu, sigma):
  part_1=(2*sigma*math.sqrt(2*math.pi))**-1
  part_2=math.exp((-(x-mu)**2) / 2*sigma**2)
  return part_1 * part_2  

my_normal_pdf(3, 3, 1)  
  
def normal_pdf(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
# 파라미터
mu = 0
sigma = 1
x = 1
# 확률밀도함수 값 계산
pdf_value = normal_pdf(x, mu, sigma)
print("정규분포 확률밀도함수 값은:", pdf_value)
                            
def my_f(x, y, z):
  return(x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
my_f(2, 9, math.pi/2)
                                             
def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)
my_g(math.pidef fname(input):
    contents
return   

# ctrl +shift +c: 커멘트 처리
#! pip install numpy
import pandas as pd
import numpy as np

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

a
type(a)

a[3]
a[2:]
#인덱싱 해도 다른 데이터 타입으로 변하지 않고 그대로 np.array로 변환되는구나!
a[1:4]

#빈 배열 채우기
b = np.empty(3)
b
b[0] = 1
b[1] = 2
b[2] = 3
b
b[2]

vec1=np.array([1,2,3,4,5])
vec1=np.arange(100)
vec1 = np.arange(1, 100.1, 0.5)
vec1

#n개로 떨어지게끔 설정
l_space1 = np.linspace(0, 1, 5)
l_space1

#0~1까지 5개의 요소, 1은 포함하지 않는 배열
linear_space2 = np.linspace(0, 1, 5, 
                            endpoint=False)
linear_space2                    
?np.linspace

# -100부터 0까지
vec2 = np.arange(0, -100, -1)
vec2
vec3 = -np.arange(0,100)
vec3

#repeat vs tile
# 차이점:repeat은 문자 그대로 반복하는 거고 
#        tile은 형식 유지한 채로 반복 
vec1=np.arange(5)
np.repeat(3, 5)
np.tile(vec1, 3)

vec1 * 2 
vec1 +vec1 
vec1 = np.array([1,2,3,4])
vec1 + vec1

max(vec1)
sum(vec1)

#35672 이하 홀수들의 합은?
x = np.arange(1, 35673, 2)
x.sum()

sum(np.arange(1, 35673, 2))

len(x)
x.shape

# 다차원 배열의 길이 재기
b = np.array([[1,2,3], [4,5,6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수

a=np.array([1,2])
b=np.array([1,2,3,4])
a + b

np.tile(a, 2)
np.repeat(a, 2) + b

b == 3

# 10 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?

# x = np.arange(0, 35672, 1)
# x
# np.tile(x, x%7=3)

#정답
sum((np.arange(1, 10) % 7 ) == 3)
# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
sum((np.arange(1, 35672) % 7 ) == 3)

# 07.15
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape
b.shape
#하나 shape되고 하나는 shape되지 않아야 브로드캐스팅이 가능하다.

#2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])

matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape
#벡터가 맞지 않아서 세로 벡터로 변경해줘야함.
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
#값이 하나인 경우 튜플 형식이 (3,) 이다 따라서
a = 3,
a
#reshape으로 변경해줌. 모양을 다시 만들어주라는 뜻.
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector

vector_1 = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector_1

vector.shape
result = matrix + vector

# (4,3) + (3,) o
# (4,3) + (4,) x
# (4,3) + (4,1) o

