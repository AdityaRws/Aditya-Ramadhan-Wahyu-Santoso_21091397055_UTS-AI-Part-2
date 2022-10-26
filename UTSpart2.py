#multiple neuron batch and multiple layer

#inisialisasi numpy
import numpy as np

#inisialisasi variabel
#Input layer feature 10 
#per batch nya 6 input
inputs = [[6.0, 1.5, 3.4, 2.7, 7.4, 2.4, 8.0, 4.9, 8.1, 1.7],
[6.3, 7.8, 6.7, 5.9, 1.3, 5.3, 2.9, 1.4, 3.9, 1.3],
[2.1, 1.1, 4.3, 5.1, 3.6, 7.2, 3.0, 6.0, 5.8, 7.0],
[0.7, 9.1, 2.3, 1.3, 8.6, 4.8, 1.3, 2.7, 6.2, 3.1],
[1.5, 7.3, 2.5, 4.0, 1.0, 4.3, 7.5, 2.8, 2.0, 6.2],
[0.6, 0.1, 0.8, 5.0, 3.4, 2.2, 2.9, 9.0, 1.3, 9.7]]

#Neuron 5 (layer1)
weights1 = [[0.3, 1.5, -3.1, 4.1, 1.37, -3.32, 7.5, -3.0, 4.8, -1.6],
[1.2, -1.0, 6.5, 3.1, -6.4, 4.1, -0.7, 0.9, -1.6, 1.5],
[-6.51, -6.0, 1.5, 4.5, -1.1, -6.2, 8.3, -3.5, 7.0, 1.0],
[-8.05, 2.15, -1.58, -4.35, 7.85, -4.0, 2.5, 0.1, -0.17, 0.34],
[0.5, 1.0, 7.4, 1.5, 4.1, -2.5, 6.1, 5.1, -5.2, 0.85]]

#banyak bias tergantung dari berapa banyak neuron pada weight1 yang ada
bias1 = [1.0, 0.45, 5.15, 1.5, 7.0]

#Neuron 3 (layer2)
weights2 = [[1.53, 4.15, 5.45, 1.7, 1.45],
[6.0, 1.37, 2.5, 5.0, 9.5],
[1.12, 7.82, 8.0, 6.0, 4.5]]

#banyak bias tergantung dari berapa banyak neuron pada weight2 yang ada
bias2 = [-5.5, 3, 2]

#output dari rumus numpy
#menghitung layer1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + bias1

#menghitung layer2 berdasarkan hasil dari perhitungan dari layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + bias2

#print output layer2
print(layer2_outputs)