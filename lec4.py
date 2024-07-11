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

