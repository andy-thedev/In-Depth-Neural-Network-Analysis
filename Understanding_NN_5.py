import numpy as np
# Using OOP to code an arbitrary neural network (forward-pass)

# For consistency in output, while achieving "randomness"
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


# We will finally begin to use objects to code our layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Unlike the previous file, we order the weights in the shape (input, neurons), so we do not have to
        # transpose in the forward-pass, for cleanliness and code efficiency
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # The first parameter for np.zeros() is the shape, so we pass in a tuple
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Layer_Dense(# features in each sample, # of neurons)
layer1 = Layer_Dense(4, 5)
# The input of our layer 2 is the # of neurons from layer 1, intuitively
# The # of neurons are chosen arbitrarily, as in layer 1
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
