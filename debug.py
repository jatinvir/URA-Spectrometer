import scipy.io

calibration_matrix = scipy.io.loadmat("Matrix_calabration_Dis7.mat")['mat2']
#matrix for validation set
probe_matrix = scipy.io.loadmat("Matrix_probe_Dis7.mat")['mat2']

print(calibration_matrix[:5])
print(probe_matrix[:5])