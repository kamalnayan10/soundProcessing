"""
First neuron implementation from scratch in python

take input -> inputs and weights
calculate sum = summation(inputs*weights)
calulate activation function(sigmoid function)

get output

"""


import math

def sigmoid(h:int):
    y = 1/(1+math.exp(-h))
    return y

def activate(inputs:list[int] , weights:list[int]):
    h = 0
    for i,j in zip(inputs,weights):
        h += i*j
    
    return sigmoid(h)


if __name__ == "__main__":
    
    inputs = [0.5, 0.3, 0.2]

    weights = [0.4, 0.7, 0.2]

    output = activate(inputs , weights)

    print(output)