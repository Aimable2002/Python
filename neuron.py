import numpy as np
import pandas as pd

inputs = [[1,2,3,2.5], [2,5,-1,2], [-1.5,2.7,3.3,-0.8]]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SimpleTransformerNeuron():
    def __init__(self, n_inputs, n_heads):
        self.n_inputs = n_inputs
        self.n_heads = n_heads
        
        # Weights for Query, Key, Value for each head
        self.w_query = 0.10 * np.random.randn(n_inputs, n_inputs)
        self.w_key = 0.10 * np.random.randn(n_inputs, n_inputs)
        self.w_value = 0.10 * np.random.randn(n_inputs, n_inputs)
        
        self.bias = np.zeros((1, n_inputs))
    
    def attention(self, inputs):
        # Create Query, Key, Value
        query = np.dot(inputs, self.w_query)
        key = np.dot(inputs, self.w_key)
        value = np.dot(inputs, self.w_value)
        
        # Attention scores
        scores = np.dot(query, key.T) / np.sqrt(self.n_inputs)
        attention_weights = softmax(scores)  # Using our custom softmax function
        
        # Weighted sum
        output = np.dot(attention_weights, value) + self.bias
        return output
    
    def forward(self, inputs):
        return self.attention(inputs)

neuron = SimpleTransformerNeuron(4, 3)
output = neuron.forward(inputs)
print(output)

