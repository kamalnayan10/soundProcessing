"""
Implementing a MLP NN(Multi Level Perception Neural Network) with forward propagation
"""

import numpy as np

class MLP:
    def __init__(self, num_inputs = 3, num_hidden = [3,5], num_outputs = 2):
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        #inititate random weights

        self.weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i] , layers[i+1])
            self.weights.append(w)

    def forward_propagate(self , inputs):
        
        activations = inputs

        for w in self.weights:
            # Calculate the net inputs
            net_inputs = np.dot(activations , w)
            
            # Calculate the activation
            activations = self.sigmoid(net_inputs)

        return activations

    def sigmoid(self , x):
        return 1/(1+np.exp(-x))

if __name__ == "__main__":
    # Create a MLP
    mlp = MLP()

    #Create inputs
    inputs = np.random.rand(mlp.num_inputs)

    #perform forward propagation
    outputs = mlp.forward_propagate(inputs)

    #print the results
    print(f"The NN inputs are: {inputs}")
    print(f"The NN output is: {outputs}")
