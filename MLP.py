import numpy as np
from random import random
import matplotlib.pyplot as plt
from datetime import datetime
# np.random.seed(0)
np.random.seed(95)


class Neural_Network(object):
  

    def __init__(self, inputs=7, outputs=1, hidden_layers=[3, 1]):
        
        ''' Constructor that takes the number of inputs, hidden layers and number of outputs'''

        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.outputs = outputs

        # layers are created by adding size of all attributes
        layers = [inputs] + hidden_layers + [outputs]

        
        # activations list created with zeros added to each index
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        
        
        # weights are randomly created bwlpw
        new_weights = []
        # derivatives list created with zeros added to each index
        derivatives = []
        for i in range(len(layers) - 1):
            
            weight = np.random.rand(layers[i], layers[i + 1])
            d = np.zeros((layers[i], layers[i + 1]))
            
            new_weights.append(weight)
            derivatives.append(d)
            
        self.activations = activations
        self.weights = new_weights
        self.derivatives = derivatives
    

    def forward_propagation(self, inputs):
        '''This method goes about the process of forward propogation that returns predicted output'''

        # the input passed to the the method is added
        activations = inputs
        self.activations[0] = activations
        
        for i, w in enumerate(self.weights):
            
            # dot product of inputs and weights
            net_inputs = np.dot(activations, w)

            # Calling of activation function (i.e. tanh)
            activations = self._tanh(net_inputs)
            self.activations[i + 1] = activations

        return activations


    def back_propagation(self, error):
        '''This method goes about the process of back propogation'''
        
        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            
            # Calling of derivative of activation function (i.e. tanh)
            delta = error * self._tanh(activations, deriv = True)        
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogation of the next error
            error = np.dot(delta, self.weights[i].T)

    
    def gradient_descent(self, learningRate=0.03, momentum = 0.6):
        '''This method goes about the process of implementing gradient descent'''
        
        # this loop updates the weights by changing gradient constantly
        for i in range(len(self.weights)):
            new_weights = self.weights[i]
            derivatives = self.derivatives[i]
            new_derivative = self.derivatives[i-1]
        
            new_weights+= derivatives * learningRate
            
            # MOMENTUM OPTIMIZER 
            
            new_weights = new_weights - (derivatives * learningRate) + (momentum * new_derivative)

    # Activation functions -> when deriv = True, the derivatives of the functions are returned

    def _sigmoid(self, x, deriv = False):

        if deriv == True:
            return x * (1.0 - x)

        
        return 1.0 / (1 + np.exp(-x))
    
    def _tanh(self, x, deriv = False):
        
        if deriv == True:
            return (1.0 - (x**2))
        return np.tanh(x)

    
    def _softmax(self, x, deriv = False):
        
        if deriv == True:
            e_x = np.exp(x)
            return e_x / e_x.sum()
        else:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()


    def _mse(self, target, output):
        """ Function that goes about conducting Mean Squared Error """
        
        # calculates subtraction of target and output
        return np.average((target - output) ** 2)

    def train(self, inputs, targets, val_inputs, val_outputs, epochs, learning_rate):
        """ Function that trains the algorithm """
        
        y_args = []
        y_args2 = []
        
        for i in range(epochs):
            sum_errors = 0
            sum_errors2 = 0
            
            # In the first loop, training set error is taken
            for j, input in enumerate(inputs):
                target = targets[j]

                # Output is taken by running forward propogation and error is calculated
                
                output = self.forward_propagation(input)
                error = target - output
                self.back_propagation(error)
                self.gradient_descent(learning_rate)

                # adding all outputs of MSE function to use later to calculate error
                sum_errors += self._mse(target, output)

            # After epoch is completed, training error is reported
            error = sum_errors / len(items)
            y_args.append(sum_errors / len(items))
      
            
            # In the second loop, validation set error is taken
            for j, input in enumerate(val_inputs):
                
                target = val_outputs[j]
                output = self.forward_propagation(input)
                error2 = target - output
                sum_errors += self._mse(val_outputs, output)

           
            error2 = sum_errors / 290
   
            # Error rate for both training and validation set is displayed
    
            print("Error: {}, Validate Error: {} at epoch {}".format(round(error,6), round(error2,6), i+1))

            y_args2.append(sum_errors / len(items))
            

        plt.plot(y_args)
        plt.plot(y_args2)
        plt.legend(['Training Set', 'Validation Set'])
        plt.savefig('MSE.png', transparent = True)
        plt.show()


        print("Training complete!")
        print("=====")
        
        
    # Automatically runs when code is run

if __name__ == "__main__":
    
    # training data
    items = np.array((input_list))
    targets =  np.array((output_list))
    
    # validation data
    items2 = np.array((val_input_list))
    targets2 =  np.array((val_output_list))
    
    # training parameters (7 inputs, 1 output and 7 neurons in one hidden layer)
    mlp = Neural_Network(7, 1, [7])
    mlp.train(items, targets, items2, targets2, 75, 0.01)

    # test data
    input = np.array(test_input_list) 
    target = test_output_list

    # returns predicted output
    output = mlp.forward_propagation(input)
    
    print("Output -> {}".format(output))
