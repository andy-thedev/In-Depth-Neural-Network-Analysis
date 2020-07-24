# Analyzing a layer from scratch

# Assume 4 arbitrarily chosen inputs
inputs = [1, 2, 3, 2.5]

# Assume the 4 inputs are fully (dense) connected to 3 output neurons. Then, each input connect to every output
# once, so there must exist three unique sets of weights that are correspondent. (4 x 3)
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# Each output neuron will have associated biases
biases = [2, 3, 0.5]

# Each output is calculated in a congruent manner to part 1. Since there are 3 outputs, there must be 3 calculations
# To influence the output, we may manipulate the weights and biases (back-propagation)
# (Since inputs are pre-destined from features, such as sensors and actual data, or outputs from the previous layer,
#  it is intuitively not a good idea to manipulate the input values)
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + biases[0],
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + biases[1],
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + biases[2]]

print(output)
