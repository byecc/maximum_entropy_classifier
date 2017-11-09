import numpy as np
import time
import math

matrix = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
sum = np.array([0,0,0])
print(sum,matrix[0])
for i in range(len(matrix)):
    sum+=matrix[i]
print(sum)

print(3*matrix[0])

print(-1*math.log2(max([-0.11,0.23,0.34,0.567])))

# print(1e-08*0.1)
# matrix = np.zeros(matrix.shape)
# print(matrix)
#
# starttime = time.time()
# for i in range(85):
#     ma = np.random.random((240000,5))
#     ma = 0.01*ma
#     # print(0.01*ma)
# print(time.time()-starttime)
