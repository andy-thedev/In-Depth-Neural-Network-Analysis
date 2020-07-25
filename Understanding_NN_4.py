import numpy as np
# Utilizing, while understanding the use of batches of inputs for neural networks

# Reasons for batches:
# 1) Batches of inputs allow us to train networks in parallel, simultaneously. This is also one of the reasons why
#    we tend to train networks on gpu instead of cpu, since cpu on average have 4~8 cores, while gpu may have thousands
#    of cores, hence, the training is much faster and efficient
# 2) Batches help with "generalization". We may classify the types of inputs, and separate them in batches, so the
#    network has less redundancy, repetition, and it may also be more helpful in assessing the network's performance,
#    competency, and loss. (Note: setting the batch size to be too large may result in over-fitting)

# 3 x 4 matrix (batch of inputs)
# This would be a potential example of three samples with four features each
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# 3 x 4 matrix
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# 3 x 1 vector
biases = [2, 3, 0.5]

# 3 x 3 matrix
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

# 3 x 1 vector
biases2 = [-1, 2, -0.5]

# It is clear that multiplication of inputs and weights, regardless of order, will result in a shape error
# Therefore, it is intuitive that we transpose the weight to make matrix multiplication possible
# (Note: we must transpose the weights rather than the inputs, since the array of inputs is made up of lists
#  from different samples. Transposing the inputs would mean that we are not multiplying each value in a sample
#  by its associated weight for every neuron)
# Although lists naturally transform to numpy arrays in the process of calculating the dot product through the
# np.dot function, we would like to convert one of them first in the parameter, to utilize the transpose function.
# If we chose not to use the numpy function, we would simply iterate through each element and transpose them manually
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# Utilize layer 1's outputs as inputs for second layer with corresponding weights and biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
