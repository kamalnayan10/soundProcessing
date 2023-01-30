"""
Implementing a MLP NN(Multi Level Perception Neural Network) with forward propagation
"""

import numpy as np
from random import random

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

        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        derivatives = []

        for i in range(len(layers) - 1):
            d = np.zeros((layers[i] , layers[i+1]))
            derivatives.append(d)

        self.derivatives = derivatives



    def forward_propagate(self , inputs):
        
        activations = inputs
        self.activations[0] = inputs

        for i,w in enumerate(self.weights):
            # Calculate the net inputs
            net_inputs = np.dot(activations , w)
            
            # Calculate the activation
            activations = self.sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations



    def back_propagate(self , error , verbose = False):
        """
            FORMULAS USED

            dE/dW_[i] = (y - a_[i+1]) s'(h_[i+!]) a_[i]

            s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))

            s(h_[i+1]) = a_[i+1]


            dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_[i] s'(h_[i]) a_[i-1]
         """

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error*self.sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0] , -1).T

            cur_activations = self.activations[i]
            cur_activations_reshaped = cur_activations.reshape(cur_activations.shape[0] , -1)

            self.derivatives[i] = np.dot(cur_activations_reshaped,delta_reshaped)

            error = np.dot(delta , self.weights[i].T)

            if verbose:
                print(f"Derivatives for W{i} : {self.derivatives[i]}")
        
        return error



    def gradient_descent(self , learning_rate):
        for i in range(len(self.weights)):
            
            weights = self.weights[i]

            derivatives = self.derivatives[i]

            weights += derivatives*learning_rate



    def train(self , inputs , targets , epochs , learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input , target in zip(inputs , targets):
                # forward propagation
                output = self.forward_propagate(input)

                # calculating error
                error = target - output

                # Back propagtion
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)
                
                sum_error += self.mse(target , output)
            
            # report error
            print(f"Error: {sum_error/len(inputs)} at epoch {i}")


    def sigmoid(self , x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self , x):
        return x*(1-x)

    def mse(self , target , output):
        return np.average((target - output)**2)


if __name__ == "__main__":
    # create MLP
    mlp = MLP(2,[5],1)
    
    # Dummy training data
    inputs = np.array([[random()/2 for i in range(2)] for i in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    

    # Train MLP
    mlp.train(inputs , targets , 50 , 0.1)

    # Create ummy data testing
    input = np.array([0.3,0.1])
    target = np.array([0.4])
    
    #Prediction   
    output = mlp.forward_propagate(input)

    print(f"Our Network believes that {input[0]} + {input[1]} = {output}")