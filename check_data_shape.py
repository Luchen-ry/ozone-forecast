import numpy as np

data = np.load('matrix_N95/data.npy')
print('data.npy shape:', data.shape)

s_matrix = np.load('matrix_N95/S_matrix.npy')
print('S_matrix.npy shape:', s_matrix.shape)

time_index = np.load('matrix_N95/time_index.npy')
print('time_index.npy shape:', time_index.shape)
print('time_index[:5]:', time_index[:5])

trainX = np.load('matrix_N95/trainX.npy')
print('trainX.npy shape:', trainX.shape)

trainY = np.load('matrix_N95/trainY.npy')
print('trainY.npy shape:', trainY.shape)