# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()
print("빈 리스트 1:", empty_list1)
print("빈 리스트 2:", empty_list2)

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list

range_list[3] = "LS  빅데이터 스쿨"
range_list[3]
# 리스트는 문자열 데이터 들어가도 괜찮음.

#두번째 원소에 
range_list[1] = ["1st", "2nd", "3rd"]
range_list

#3rd만 가져오고 싶다면?
range_list[1][2]

# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져 있다 -> 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares

# 3, 5, 2, 15의 3제곱
my_squares=[x**3 for x in [3, 5, 2, 15]]
my_squares

# numpy array(배열)이 와도 가능
import numpy as np
my_squares=[x**3 for x in np.array([3, 5, 2, 15])]
my_squares

import pandas as pd

# pandas 시리즈 와도 가능!
exam = pd.read_csv("data/exam.csv")
my_squares=[x**3 for x in exam["math"]]
my_squares

# 리스트 합치기
3 + 2
"안녕" + "하세요"
"안녕" * 3
# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)

#x가 numbers로 들어가면 계속 그 숫자 x를 (0,1,2) 세번 씀 
# x는 numbers에 있는 첫번째인 5를 가져와서 쓸건데 range(3)번 (0,1,2) 총 세번 쓸거임. 계속 반복
numbers = [5, 2, 3]
repeated_list = [ x for x in numbers for _ in range(3)]
repeated_list

numbers = [5, 2, 3]
repeated_list = [ x for x in numbers for _ in [4, 2, 1, 10]] # 길이 4개짜리 원소임
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list

# _ 의 의미 (참고, 알면 좋음.)
# 앞에 나온 값을 가리킴
5 + 4
_ + 6    # _는 9를 의미

# 값 생략, 자리 차지 (placeholder)
a, _, b = (1, 2, 4)
a; b
_
_ = None
_
del _

# for 루프 문법
# for x i in 방위:
#작동방식
for x in [4, 1, 2, 3]:
    print(x)

for i in range(5):
    print(i**2)

# 리스트를 하나 만들어서 
# for 루프를 사용해서 2 , 4, 6, 8,..., 20의ㅣ 수를 
# 채워넣어보세요!
# [i for i in range(2, 21, 2)]

# append: 주어진 리스트 맨 마지막 자리에 들어온 숫자를 집어넣음.
mylist = []
mylist.append(2)
mylist.append(4)
mylist.append(6)
mylist

for i in range(1, 11):
    mylist.append(i*2)
mylist

mylist = [0] * 10
for i in range(10):
    mylist[i] = 2 * (i + 1) 
mylist

# 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 10

for i in range(10):
    mylist[i] = mylist_b[i]
mylist
#이건 머임
mylist = list(range(1, 11))
mylist

# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

for i in range(5):
    mylist[i] = mylist_b[i*2]
mylist

# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서.
# for 루프의 :는 생략한다.
# 실행부분을 먼저 써준다.
# 결과값을 발생하는 표현만 남겨두기
[i*2 for i in range(1, 11)]
[x for x in numbers]


#[0,1,2] = range(3)
# i 위쪽단계 고정된채로 두번째 j 루프 다 돌아감.
for i in [0, 1]:
    for j in [4, 5, 6]:
            print(i, j)
            
            
for i in [0, 1]:
    for j in [4, 5, 6]:
            print(i)

numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i, j)

numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i)

# 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]

# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

# 원소체크
# [x == "banana" for x in fruits]
mylist=[]
for x in fruits:
    mylist.append(x == "banana")
mylist    

#바나나의 위치를 뱉어내게 하려면?
fruits = ["apple", "apple", "banana", "cherry"]

import numpy as np
#리스트 형태를 어레이로 바꿔주기
fruits = np.array(fruits)
int(np.where(fruits == "banana")[0][0])
# 1차원 넘파이 어레이 -> 넘파이 인티저를 다시 인티저로 바꾸기
# int() 이유는 인트를 또 인트로 바꿔주기

# 원소 거꾸로 써주는 reverse()
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse()
fruits

# 원소 맨끝에 붙여주기
fruits.append("pineapple")
fruits

# 원소 삽입
fruits.insert(2, "test")
fruits

# 원소 제거
fruits.remove("test")
fruits

fruits.remove("apple")
fruits

import numpy as np
# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])
# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)
# ~ 논리형이 반대가 됨 바나나와 애플이 아닌것들이 true가 됨
~np.isin(fruits, items_to_remove)
# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)

#교재 뒤쪽은 나중에 보기





























