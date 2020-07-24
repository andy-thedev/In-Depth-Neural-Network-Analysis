import numpy as np
# Cleaning up code from the previous analysis through loops/numpy

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
Output of current layer
layer_outputs = []

The zip() function combines two lists into a zip object, which is an iterator of tuples, element-wise
ie: ([0.2, 0.8, -0.5, 1.0], 2) , ([0.5, -0.91, 0.26, -0.5], 3), ...

for neuron_weights, neuron_bias in zip(weights, biases):
    # Output of given neuron
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
'''

# Using the numpy dot product function for the above, commented algorithm instead
# (Note: The two lists, weights and inputs are converted to numpy arrays in the process of calculating the dot product)
# Since the shape of our weights is 3 x 4 and the shape of our inputs is 4 x 1 (vectors are classified as (4,)),
# by matrix multiplication properties, we must have weights as our first parameter to receive a return shape of
# 3 x 1, and avoid shape errors.
# Furthermore, the three sets of weights suggest there are three output neurons, intuitively.
output = np.dot(weights, inputs) + biases

print(output)
