import numpy as np

#def read_matrix_from_file(filename):
#    matrix = []
#    with open(filename, 'r') as file:
#        for line in file:
#            #row = line.strip().split()
#            row = list(line.strip())  # 各行を1文字ずつリストに
#            matrix.append(line)
#    return np.array(matrix)

# 使用例
filename = 'nyanko.txt'  # 読み込むファイル名
matrix = np.loadtxt(filename, dtype=int)
#print(matrix)
#print(np.shape(matrix))

num_neurons_percol, num_neurons_perrow = np.shape(matrix)

#print(num_neurons_percol)
#print(num_neurons_perrow)

network_params = {
    'num_neurons_percol': num_neurons_percol,
    'num_neurons_perrow': num_neurons_perrow,
    'num_neurons': num_neurons_percol * num_neurons_perrow,
}
