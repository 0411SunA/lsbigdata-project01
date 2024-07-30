import numpy as np

matrix = np.vstack((np.arange(1, 5),
                    np.arange(12, 16)))
print("행렬:\n", matrix)                    

np.zeros(5)
np.zeros([5, 4])
np.arange(1, 7).reshape((2,3))
np.arange(1, 7).reshape((2,-1))

# 0에서 99까지의 수 중 랜덤 50개 뽑아서 5 by 10 행렬 만들어라
np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape(5, -1)
a

mat_a = np.arange(1, 21).reshape((4,5), order="F")
mat_a

#인덱싱
mat_a[0, 0]
mat_a[1, 1]
mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

# 행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]

# 짝수 행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1::2,:]
mat_b[[1, 4, 6, 14],]

x = np.arange(1, 11).reshape((5,2)) * 2
x
x[[True, True, False, False, True], 0]

mat






