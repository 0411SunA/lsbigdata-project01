import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
# seed = random값 고정해주는 역할 함.
np.random.seed(2024)
a = np.random.randint(1, 21, 10)

print(a)
a = np.random.choice(np.arange(1, 21), 10, False)
print(a)
# 함수정보 알고싶을 때 ? 함수 붙이기 -> help 창 보기
# 시뮬레이션 다른사람에게 결과 공유할 때
# 두 번째 값 추출
print(a[1])

# 빈도수 임의로 설정하는 법.
a = np.random.choice(np.arange(1, 4), 100, True, np.array([2/5, 2/5, 1/5]))
print(a)
sum(a == 1)
sum(a == 2)
sum(a == 3)


a[2:5]
a[-1]
a[-2] #맨 끝에서 두번째
a[0:6:2] #start:stop:step 자리 -> 한칸 띄고 결과 나옴
#1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(1, 1000) % 3 = 0)
#정답 하나
sum((np.arange(3, 1001, 3))
#다른 정답
x = np.arange(1, 1001)
sum(x[2:1000:3])
# 다른 정답
x = np.arange(0, 1001)
sum(x[::3])

x = np.arange(0, 1001)
x
#첫번째, 세번째, 다섯번째 값 추출
print(a[[0,2,4]])
#4번째 값 제거
np.delete(a, 3)
#리스트 사용해서 멀티 인덱스 표현할 수 있음.
np.delete(a, [1,3])

a > 3
a
a[a>3]
b = a[a>3]
print(b)
#원소들 필터링할 때 논리형 많이 쓰임. true, false

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a > 2000) & (a < 5000)]
a > 2000
a < 5000
# a[조건을 만족하는 논리 연산자]
(a > 2000) & (a < 5000)


#!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
np_df=np.array(df['mpg'])

row_names = np.array(df['mpg'])
model_names = np.array(df.index)
model_names

# 15 이상 25이하인 데이터 개수는?
sum(((np_df>= 15) & (np_df <= 25)))
model_names[(np_df>= 15) & (np_df <= 20)]

#연습
sum(((np_df>=15) & (np_df<=25)))

# 평균 mpg 보다 높은 (이상) 자동차 대수는?
sum(np_df >= np.mean(np_df))
# "" 낮은 모델은 ''? : 연비 좋지 않은 차.
model_names[np_df < np.mean(np_df)]


# 15보다 작거나 22 이상인 데이터 개수는?
sum(((np_df < 15) | (np_df >= 22)))

np.random.seed(2024)
#1부터 10000까지 5개 랜덤 숫자 추출
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
#a[조건을 만족하는 논리형벡터]
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]
#필터링해서 값 부여하기 vip 부여 등 할 수 있음.
a[a > 3000] = 3000
a
b

np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a < 50
np.where(a < 50) #true 있는 위치를 반환한다.

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a
#처음으로 5000보다 큰 숫자가 나왔을 때, 
#숫자 위치와 그 숫자는 무엇인가요?
a > 5000
a[0]
#정답
a[a > 5000][0]
a[np.where(a > 5000)][0]
#10,000 보다 큰 숫자
a[np.where(a>10000)][0]
a[np.where(a>10000)]
#위치
np.where(a > 10000)
# 22000보다 큰 숫자
x = np.where(a > 22000)
x
type(x) #튜플인 이유: 괄호로 끝남. 또한 ,)로 끝난다. -> 원소 하나임.
# 튜플에서 첫번째 원소를 뽑아야겠다 [0]
# 원소 1번째인 숫자 10을 꺼내야해서 두번째에 [0] 쓴거임.
my_index = x[0][0]
a[my_index]
a[np.where(a > 22000)][0]

#처음으로 24000보다 큰 숫자 나왔을때, 위치와 그 숫자는?
a[np.where(a >24000)][0]
x = np.where(a > 24000)
my_index = x[0][0]
a[my_index]

#처음으로 10000보다 큰 숫자들 중
# 50번째로 나오는 숫자 위치와 그 숫자는 무엇인가
x = np.where(a > 10000)
x[0][49]
a[x[0][49]]
a[np.where(a>10000)][49]
# 500보다 작은 숫자들 중 
# 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가
x = np.where(a < 500)
#숫자 위치
x[0][-1]
#그 숫자 구하기
a[x[0][-1]]

a = np.array([20, np.nan, 13, 24, 309])
a + 3
np.mean(a)
#nan 무시하고 나머지 숫자의 평균 구하기
np.nanmean(a)
#nan을 원하는 값으로 변경하고 싶을 때 사용, 외우진 말고 이런게 있다!정도 알아라
np.nan_to_num(a, nan = 0)

False
a = None
b = np.nan
a
b
#a는 계산도 안됨.
a + 1
b + 1

~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

str_vec = np.array(["사과", "배", "수박", "참외"], dtype = str)
str_vec
str_vec[[0,2]]

#숫자가 문자열 벡터로 바뀜.
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

#튜플로 묶음 numpy array로 이루어진 리스트건 튜플이건 결과는 numpy array로 나옴.
combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

#벡터들 묶어주자
col_stacked = np.column_stack((np.arange(1, 5),
                               np.arange(12, 16)))
col_stacked
#vstack
row_stacked = np.vstack((np.arange(1, 5),
                               np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5),
                                  np.arange(12, 18)))
uneven_stacked
# 길이가 다른 벡터
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 30)

#재활용개념: vec2 길이만큼 출력하라는 뜻.
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked

#연습문제 각 원소에 5 더하기
a = np.array([1, 2, 3, 4, 5])
a + 5
#문제 2.홀수 번째 요소만 추출
a = np.array([12, 21, 35, 48, 5])
a[0::2]
#최대값 찾기
a = np.array([1, 22, 93, 64, 54])
a.max()
#중복값 제거 -> np.unique()
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)
# 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요.
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b
x = np.empty(6)
x
#홀수
x[[1,3,5]] = b
x
#짝수 x 짝수로 들어감
x[[0, 2, 4]] = a
x

#다른 방법
x[0::2] = a
b
x[1::2] = b
x
