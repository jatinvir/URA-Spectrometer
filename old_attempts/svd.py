import scipy.io

data = scipy.io.loadmat("Matrix_calabration_Dis7.mat")

mat2 = data['mat2']

print(mat2.shape)
print(mat2.dtype)
print(mat2)