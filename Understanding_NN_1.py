import numpy as np
# Analyzing a singular "connection" from scratch

# Assume we receive 4 outputs from the previous layer of neurons as inputs for our building layer
inputs = [1, 2, 3, 2.5]
# Assume the following are the weights for the respective inputs above
weights = [0.2, 0.8, -0.5, 1.0]
# There are 4 connections, but only one neuron, hence the singular bias
# To further analyze, we have 4 inputs, and 4 weights associated, which connect to a singular neuron, holding
# the corresponding bias (+: adding an input does not increase the number of biases)
bias = 2

'''
The multiplicative form is of dot product
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
'''

# Utilizing the numpy dot product function instead:
output = np.dot(weights, inputs) + bias


print(output)