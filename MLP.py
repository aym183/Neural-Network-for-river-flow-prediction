import numpy as np
from random import random
import matplotlib.pyplot as plt
from datetime import datetime
# np.random.seed(0)
np.random.seed(95)


class Neural_Network(object):
  

     def __init__(self, inputs, outputs, hidden_layers):
        
        ''' Constructor that takes the number of inputs, hidden layers and number of outputs'''

        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.outputs = outputs

        # layers are created by adding size of all attributes
        layers = [inputs] + hidden_layers + [outputs]

        
        # activations list created with zeros added to each index
        activ_operations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activ_operations.append(a)
        
        
        # weights are randomly created bwlpw
        new_weights = []
        # derivatives list created with zeros added to each index
        deriv_operations = []
        for i in range(len(layers) - 1):
            
            weight = np.random.rand(layers[i], layers[i + 1])
            d = np.zeros((layers[i], layers[i + 1]))
            
            new_weights.append(weight)
            deriv_operations.append(d)
            
        self.activ_operations = activ_operations
        self.weights = new_weights
        self.deriv_operations = deriv_operations
    

    def forward_propagation(self, inputs):

        '''This method goes about the process of forward propogation that returns predicted output'''

        # the input passed to the the method is added
        activ_operations = inputs
        self.activ_operations[0] = activ_operations
        
        for i, w in enumerate(self.weights):
            
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activ_operations, w)

            # Calling of activation function (i.e. tanh)
            activ_operations = self._sigmoid(net_inputs)
            
            self.activ_operations[i + 1] = activ_operations

        return activ_operations


    def back_propagation(self, error):
        '''This method goes about the process of back propogation'''
        
        for i in reversed(range(len(self.deriv_operations))):

            activ_operations = self.activ_operations[i+1]
            
            # Calling of derivative of activation function (i.e. tanh)
            delta = error * self._sigmoid(activ_operations, deriv = True)

        
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activ_operations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.deriv_operations[i] = np.dot(current_activations, delta_re)

            # backpropogation of the next error
            error = np.dot(delta, self.weights[i].T)

    
    def gradient_descent(self, learningRate=0.03, momentum = 0.6):
        '''This method goes about the process of implementing gradient descent'''
        
        # this loop updates the weights by changing gradient constantly
        for i in range(len(self.weights)):
            new_weights = self.weights[i]
            deriv_operations = self.deriv_operations[i]
            new_derivative = self.deriv_operations[i-1]
        
            new_weights+= deriv_operations * learningRate
            
            # MOMENTUM OPTIMIZER -> used in final algorithm (mentioned why in report)
            
            new_weights = new_weights - (deriv_operations * learningRate) + (momentum * new_derivative)

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
        
        
    

    def train(self, inputs, targets, val_inputs, val_outputs, test_inputs, test_outputs, epochs, learning_rate):
        """ Function that trains the algorithm """
        
        y_args = []
        y_args2 = []
        
        destand_outputs = []
        test_destand_outputs = []
        train_destand_outputs = []
        
        # Destandardizing of outputs of training, validation and test sets
        for i in range(len(val_inputs)):
                destand_outputs.append([(((val_outputs[i][0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton])
        
        for i in range(len(targets)):
                train_destand_outputs.append([(((targets[i][0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton])
                
                
        for i in range(len(test_outputs)):
                test_destand_outputs.append([(((test_outputs[i][0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton])


        # TRAINING TAKES PLACE HERE

        for i in range(epochs):
            sum_errors = 0
            sum_errors2 = 0
            
            # In each loop, training input at each index is taken
            for j, input in enumerate(inputs):
                target = targets[j]

                # Output is taken by running forward propogation and error is calculated
                
                output = self.forward_propagation(input)  

                error = target - output
                self.back_propagation(error)
                self.gradient_descent(learning_rate)

                # Destandardizing of output

                output[0] = (((output[0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton 
    
                
                # adding all outputs of MSE function to use later to calculate error with destandardized values
                sum_errors += self._mse(train_destand_outputs[j][0], output[0])

            # After epoch is completed, training error is reported
            error = sum_errors / len(items)

            # RMSE
            root_error = math.sqrt(sum_errors / len(items))
            y_args.append(sum_errors / len(items))
            sum_errors = 0
                
        
            # In each loop, validation input at each index is taken
            for j, input in enumerate(val_inputs):
                
                target = val_outputs[j]
                
                output = self.forward_propagation(input)

                # Destandardizing of output
                output[0] = (((output[0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton 
        
                sum_errors += self._mse(destand_outputs[j][0], output[0])
    
            error2 = sum_errors / 290
            y_args2.append(sum_errors / 290)
    
            sum_errors = 0
        
            # In each loop, test input at each index is taken
            for j, input in enumerate(test_inputs):
                
                target = test_outputs[j]
            
                
                output = self.forward_propagation(input)
               
                # Destandardizing of output
                output[0] = (((output[0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton 
                    
                sum_errors += self._mse(test_destand_outputs[j][0], output[0])
    
            error3 = sum_errors / 290
            
            sum_errors = 0

            # RMSE Error rate for training, validation and test set is displayed

            print("Error: {}, Validate Error: {}, Test Error: {} at epoch {}".format(round(math.sqrt(error),6),round(math.sqrt(error2),6) ,round(math.sqrt(error3),6), i+1))


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
    
    input = np.array((test_input_list))
    target = np.array((test_output_list))
    # training parameters (7 inputs, 1 output and 7 neurons in one hidden layer)
    mlp = Neural_Network(7, 1, [7])
    mlp.train(items, targets, items2, targets2, input, target, 750, 0.7)


    # returns predicted output
    output = mlp.forward_propagation(input)
    
#  Destandardizing of Expected and Predicted output to display

    new_test_output = []
    for i in range(len(output)):
        
        new_test_output.append((((test_output_list[i][0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton)
        output[i][0] = (((output[i][0] - 0.1)*(max_skelton - min_skelton))/ 0.3) + min_skelton
    

    for i in range(len(output)):
        print("Actual Output -> {} Output -> {}".format(round(new_test_output[i], 6), round(output[i][0], 6)))
        
        