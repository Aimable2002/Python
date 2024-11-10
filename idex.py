import numpy as np

from nltk.tokenize import word_tokenize




def preprocess_word(text):
    input_text = word_tokenize(text.lower())
    print(input_text)
    return input_text

text = "Hello, how are you?"

# np.random.seed(0)

# X = [[1,2,3.5], [2,3,4.4], [3,4,5.5]]


# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#         print("weights shape:", np.array(self.weights).shape)
#         self.biases = np.zeros((1, n_neurons))
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

# layer1 = Layer_Dense(3, 5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)

# print(layer1.output)

# layer2.forward(layer1.output)
# print(layer2.output)
# print("X shape:", np.array(X).shape)  






# weight = [[1,2,3],[4,5,6],[7,8,9]]
# biase = [1,2,3]

# weight2 = [[1,2,3],[4,5,6],[7,8,9]]
# biase2 = [1,2,3]

# output = np.dot(inputs, np.array(weight).T) + biase
# output = np.dot(inputs, weight) + biase


# layer_outputs = []
# for neuron_weights, neuron_biase in zip(weight, biase):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_biase
#     layer_outputs.append(neuron_output)

# print(layer_outputs)