# 소스코드 보고싶을 때
def g(x=3):
    result = x + 1
    return result

g()

print(g)

# 함수 내용확인
import inspect
print(inspect.getsource(g)) # getsource 선택 하고 F12 누르면 정보 알 수 있음

import numpy as np
np.array([1,2,3])

# if .. else 정식
x = 3
if x > 4:
    y = 1
else:
    y = 2
print(y)

# if else 축약
y = 1 if x > 4 else 2
y

# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

# 조건 3개 이상인 경우
x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)

# 여러 조건을 처리하는 numpy.select()
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices =    ["양수","0","음수"]
result = np.select(conditions, choices, x)
print(result)

# for loop
for i in range(1, 4):
    print(f"Here is {i}")

# for loop 리스트 컴프
print([f"Here is {i}" for i in range(1, 4)])

name = "남규"
age = "31 (진)"
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)

import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30]) # 나이 배열의 길이를 names 리스트와 맞춤

# zip() 함수 활용법
import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30])
# zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)
# 각 튜플을 출력
for name, age in zipped:
    print(f"Name: {name}, Age: {age}")

# while 문
i = 0
while i <= 10:
    i += 3
    print(i)

# while, break 문
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)

# apply
import pandas as pd

data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]})

data

data.apply(sum, axis=0)
data.apply(max, axis=1) # axis=1 행 기준

def my_func(x, const=3): # ex: axis=1) 4**2 + const값 = 19 됨. 이런식임
    return max(x)**2 + const

my_func([3, 4, 10], 5)

data.apply(my_func, axis=0, const=5)

# 있다 정도만 알아두면 좋음
import numpy as np
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

np.apply_along_axis(max, axis=0,
                    arr = array_2d)


# 함수 환경
# 환경이 다 다르게 설정된다고 이해하기.
# Global 환경에 존재하는 변수 y와 my_func() 안에 존재하는 y는 다른 환경에서 같은 이름을
# 가진 변수
y = 2

def my_func(x):
    global y

    def my_f(k):
        return k**2
    y = my_f(x) + 1

    result = x + y

    return result

print(y)
my_func(3)

# my_func(3) 이걸 해줘야 실질적으로 코드가 돌아간다. 
 
my_f(3) # 에러 나는 게 정상
my_func(3) # 이건 돌아감. 왜냐면 my_func() 안에 있는 함수이기 때문임! 3 ** 2 + 1 + 3

print(y) # x 3으로 받아서 y가 업데이트된게 글로벌 키로 반영되서  3**2 + 1

# 입력값이 몇 개일지 모를땐 별표를 * 붙임
def add_many(*args): 
    result = 0
    for i in args: 
        result = result + i

    return result

add_many(1, 2, 3)

def first_many(*args): 
    return args[0]

first_many(1, 2, 3)
first_many(4, 1, 2, 3)

def add_mul(choice, *my_input): 
     if choice == "add":   # 매개변수 choice에 "add"를 입력받았을 때
         result = 0 
         for i in my_input: 
             result = result + i 
     elif choice == "mul":   # 매개변수 choice에 "mul"을 입력받았을 때
         result = 1 
         for i in my_input:              result = result * i 
     return result 

add_mul("add", 5, 4, 3, 1)

# 별표 두개 (**)는 입력값을 딕셔너리로
# 만들어줌!
def my_twostars(choice, **kwargs):
    if choice == "first":
        return print(kwargs["age"])
    elif choice == "second":
        return print(kwargs["name"])
    else:
        return print(kwargs)
    
my_twostars("first", age=30, name="SUN A")    
dic_a = {'age': 30, 'name': 'sun a'}
dic_a["age"]
dic_a["name"]