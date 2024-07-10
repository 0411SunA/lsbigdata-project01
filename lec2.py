#ctrl + enter
#shift + 화살표 = 블록

a=1
a

## 파워쉘 명령어 리스트
# ls: 파일 목록
# cd: 폴더 이동
# . 현재폴더
# .. 상위폴더
# cd.. : 상위폴더로 올려줘

#show folder in new window: 해당위치 탐색기

# cls: 화면 정리
a = 10
a
#항상 오른쪽에 있는 걸 왼쪽에 넣어줘라 = 할당한다 assign
#왼쪽이 변수

a = "안녕하세요!"
a='안녕하세요!'

a= [1,2,3]
a
b = [4,5,6]
b
a+b

a='안녕하세요!'
a
b= 'LS 빅데이터 스쿨!'
b
a + b
a+ ' ' + b
# concatenate: 사슬같이 잇다. str끼리만 연결할 수 있다.

print(a)

num1 = 3
num2 = 5
num1 + num2

a = 10
b = 3.3

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a * b)
print("a / b =", a/b)
print("a % b =", a%b)
print("a // b =", a//b)
print("a ** b =", a**b)
(a ** 2) //7
(a ** 2) % 7
# shift + alt + 아래화살표: 아래로 복사
# ctrl + alt + 아래화살표: 커서 여러개

a
b
a == b
a != b
a <  b
a >  b
a <= b
a >= b

# 2에 4승과 12345을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
# 9의 7승을 12로 나누고, 36452를 253으로 나눈 나머지에 곱한수 중 큰 것은?

((2**4) + (12345 // 7) ) % 8
((9**7)%12) *(36452%253)

#정답
a = ((2 ** 4) + (12453 // 7)) % 8
b = ((9 ** 7) /12) * (36452 % 253)
a < b

user_age = 25
is_adult = user_age >=18
print("성인입니까?", is_adult)

# False = 3
# True = 2
a = "True"
TRUE = 4
b = TRUE
b
C = true
d = True

# True, False
a = True
b = False

a and b
a or b

# True: 1
# False: 0
True + True
True + False
False + False

# and 연산자
True and False
True and True
False and False
False and True

True * False
True * True
False *  False
False *  True


# or 연산자
True or True
False or False
False or True
False or False

a = False
b = False
a or b
min(a+b, 1)

# 한 번도 구매하지 않은 사람과 한번이라도 구매해본 사람 구분
# 1. or: 1~12월 구매 여부 T, F or로 묶어서 T나오면 구매한 사람, F: 구매 안한 사람
# 2. and: and 묶어서 1보다 크면 구매한 사람으로 판정
# 3. sum: sum(1~1월 구매여부) -> 1이상이면 구매 경험 있음

a = 3
#a = a + 10
a += 10
a

a -= 4
a %= 3
a
a +=12
a **= 2
a /= 7

str1 = "hello"
str1 + str1
str1*3

str1 = "Hello!"
# 문자열 반복
repeated_str = str1 * 3
print("Reapeated string:", repeated_str)

str1 * 2.5

# 정수: int(eger)
# 실수: float(double)

# 단항 연산자 (심화학습 굳이 깊게 x)
x = 5
+x
-x
~x

#binary (2진수로 표현함.)
bin(5)
bin(-6)
#따옴표롭 붙어있어서 문자임
#0b가 앞에 붙어있으면 이진수다. -> 2의 0승, 1승, 2승

bin(1)
bin(-2)
x=3
~x
bin(-4)
bin(3)
~x
bin(3)

x = -4
~x
bin(-4)
bin(3)

bin(-4)
bin(~4)

bin(~0)

max(3, 4)
var1 = [1,2,3]
sum(var1)

!pip install pydataset
import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df

!pip install pandas
import pandas
!pip install numpy
import numpy
