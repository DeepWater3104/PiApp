import numpy as np

filename = 'nyanko.txt'
matrix = np.loadtxt(filename, dtype=int)

num_neurons_percol, num_neurons_perrow = np.shape(matrix)

network_params = {
    'num_neurons_percol': num_neurons_percol,
    'num_neurons_perrow': num_neurons_perrow,
    'num_neurons': num_neurons_percol * num_neurons_perrow,
}
