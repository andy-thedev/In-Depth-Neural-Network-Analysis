import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Analyzing activation functions

# Activation functions are critical in neural network model design, as the calculation of weights and biases
# can only be linear, no matter how complex the layers are. This causes loss in sophistication and accuracy, as
# problems that may require higher degrees of calculation may only be approximated, strictly through linear forms.
# Activation functions allow us to break free from this limitation, and provide a higher level of necessary complexity

# Examples of popular activation functions:

# Step function: y = 0 if x <= 0,
#                y = 1 if x > 0

# Sigmoid function: y = 1/(1+e^(-x))
# Possible con: vanishing gradient

# ReLU (Rectified Linear Unit) function: y = 0 if x <= 0
#                                       y = x if x > 0
# Possible pro: fast to calculate
# Very popular for hidden layers in neural networks
# We will analyze this specific activation function in the following code:

# Coding ReLU from scratch
# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# Method 1
'''
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
'''

# Method 2
'''
for i in inputs:
    output.append(max(0, i))
print(output)
'''

# Coding a neural network using ReLU with OOP

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# 100 feature sets, 3 classes
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Unlike the previous file, we order the weights in the shape (input, neurons), so we do not have to
        # transpose in the forward-pass, for cleanliness and code efficiency
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # The first parameter for np.zeros() is the shape, so we pass in a tuple
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Layer_Dense(Size of features in each sample, # of neurons)
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

