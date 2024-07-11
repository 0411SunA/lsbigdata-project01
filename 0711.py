# 데이터 타입
x = 15.34
print(x, "는 ", type(x), "형식입니다.", sep='')
#sep='' 입력값들 사이를 무엇으로 채울 것인가. '' 는 아무것도 넣지말라는 뜻.

# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(ml_str)
print(ml_str, type(ml_str))

#문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)

#문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)

#리스트: 숫자, 문자 모두 담을 수 있음 다른 요소들도 섞어서 가능
fruit = ["apple", "banana", "cherry"]
type(fruit)

numbers = [1, 2, 3, 4, 5]
type(numbers)

#튜플: 소괄호로 생성함.
#리스트, 튜플로 각각 만들어서 두번째 원소를 25로 변경할 것.
#리스트는 값 변경이 중간에 가능한데 튜플은 불가능한 대신 실행 시간이 빠름.
#튜플 사용할 때: 바꾸면 안되는 값 사용할 때, 휴먼에러 발생 가능성 줄이기
a_tp1 = [10, 20, 30]
a_tp2 = (10, 20, 30)
a_tp1[1] = 25
a_tp2[2] = 25
a_tp1
a_tp2

a[0]
a[1] = 25

b_int = (42)
b_int
type(b_int)
b_int = 10
b_int
b_tp = (42,)
b_tp
b_tp = 10
b_tp
type(b_tp)

a_tp = (10, 20, 30, 40, 50)
a_tp[3:] #해당 인덱스 이상
a_tp[:3] #해당 인덱스 미만
a_tp[1:3] #해당 인덱스 이상 & 미만

a_ls = [10, 20, 30, 40, 50]
a_ls[1:4]
a_ls[:3]
a_ls[2:]

# 사용자 정의함수

def min_max(numbers):
    return min(numbers), max(numbers)
a = [1, 2, 3, 4, 5]

result = min_max(a)
result
type(result)

print("Minimum and maximum:", result)

# 딕셔너리
person = {
'name': 'John',
'age': 30,
'city': 'New York'
}
suna = {
'name': 'SunA',
'age': (23, 24, 25),
'city': 'Hanam'
}
print("Person:", person)
print("Person:", suna)
#튜플로 원하는 숫자 추출하기
suna.get('age')[2]

suna_age = suna.get('age')
suna_age[0]

#집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)
type(fruit)

# 집합에서 사용가능한 메서드 요소 추가 : add
empty_set.add('apple')
empty_set.add('banana')
empty_set.add('apple')
#제거하기 discard는 요소에 집합 없어도 에러 안뜸
empty_set.remove("banana")
empty_set.discard("mango")

empty_set

#집한 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) #합집합
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다

is_active = True
is_greater = age > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

#조건문
a=3
if (a == 2):
    print("a는 2와 같습니다.")
else:
    print("a는 2와 같지 않습니다.")
    
# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

#문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'} 
#set 데이터집합: 중괄호 {}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

#자습
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

# 논리형과 숫자형 변환 예제
# 숫자를 논리형으로 변환
zero = 0
non_zero = 7
bool_from_zero = bool(zero) # False
bool_from_non_zero = bool(non_zero) # True
print("0를 논리형으로 바꾸면:", bool_from_zero)
print("7를 논리형으로 바꾸면:", bool_from_non_zero)
# 논리형을 숫자로 변환
true_bool = True
false_bool = False
int_from_true = int(true_bool) # 1
int_from_false = int(false_bool) # 0
print("True는 숫자로:", int_from_true)
print("False는 숫자로:", int_from_false)
# 논리형과 문자열형 변환 예제
# 논리형을 문자열로 변환
str_from_true = str(true_bool) # "True"
str_from_false = str(false_bool) # "False"
print("True는 문자열로:", str_from_true)
print("False는 문자열로:", str_from_false)
# 문자열을 논리형으로 변환
str_true = "True"
str_false = "False"
bool_from_str_true = bool(str_true) # True
bool_from_str_false = bool(str_false) # True, 비어있지 않으면 무조건 참
print("'True'는 논리형으로 바꾸면:", bool_from_str_true)
print("'False'는 논리형으로 바꾸면:", bool_from_str_false)

