import numpy as np

matrix = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
sum = np.array([0,0,0])
print(sum,matrix[0])
for i in range(len(matrix)):
    sum+=matrix[i]
print(sum)

print(3*matrix[0])