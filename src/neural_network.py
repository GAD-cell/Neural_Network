from math import*
from random import*
import matplotlib.pyplot as plt
import numpy as np
from src.functions import *
import pickle

class NeuralNetwork():
    """
    Neural Network implementation for deep learning tasks.
    
    This class provides a flexible implementation of a neural network with
    customizable activation functions, cost functions, and gradient descent methods.
    """

    def __init__(self, network_architecture, activation_function, cost_function, gradient_method, learning_rate):
        """
        Initialize the neural network with the specified architecture and functions.
        
        Args:
            network_architecture (list): List of integers representing the number of neurons in each layer
            activation_function (function): Activation function to use (e.g., sigmoid, ReLU)
            cost_function (function): Cost function to use (e.g., mean square error, log loss)
            gradient_method (function): Gradient descent method to use (e.g., SGD, Adam)
            learning_rate (float): Learning rate for gradient descent
        """
        self.act = activation_function  
        self.cost = cost_function             
        self.method = gradient_method 
        self.alpha = learning_rate
        self.parametres = {}
        self.C = len(network_architecture)
        for c in range(1, self.C):
            n = network_architecture[c]
            m = network_architecture[c-1]
            # Xavier/Glorot initialization for weights
            self.parametres['W'+ str(c)] = np.random.uniform(low=-sqrt(6)/sqrt(n+m), high=sqrt(6)/sqrt(n+m), size=(n, m))
            self.parametres['b'+str(c)] = np.zeros((network_architecture[c], 1))

    def feedforward(self, x):
        """
        Perform forward propagation through the network.
        
        Args:
            x (numpy.ndarray): Input data
            
        Returns:
            tuple: (activations, Z values) for all layers
        """
        activations = {'A0': x}
        dico_Z = {}
        
        for c in range(1, self.C):
            Z = np.dot(self.parametres['W' + str(c)], activations['A' + str(c-1)]) + self.parametres['b' + str(c)]
            dico_Z['Z' + str(c)] = Z
            activations['A' + str(c)] = self.act(Z)[0]
            
        return(activations, dico_Z)

    def backprop(self, input_data, target_output):
        """
        Perform backpropagation to compute gradients.
        
        Args:
            input_data (numpy.ndarray): Input data
            target_output (numpy.ndarray): Target output
            
        Returns:
            tuple: (gradients, loss value)
        """
        gradients = {}
        activations, dico_Z = self.feedforward(input_data)
        deltaZ = self.cost(activations['A' + str(self.C-1)], target_output)[1]
        loss = self.cost(activations['A' + str(self.C-1)], target_output)[0]
        for c in reversed(range(1, self.C)):
            gradients['deltaw' + str(c)] = np.dot(np.multiply(deltaZ, self.act(dico_Z['Z' + str(c)])[1]), activations['A'+str(c-1)].T)          #DL/DW
            gradients['deltab' + str(c)] = np.multiply(deltaZ, self.act(dico_Z['Z' + str(c)])[1])                                                #DL/DB
            
            deltaZ = np.dot(self.parametres['W' + str(c)].T, np.multiply(deltaZ, self.act(dico_Z['Z' + str(c)])[1]))                          #DL/DX calculated through backpropagation
        
        return(gradients, loss)
 
    def train(self, input_data, target_output, facteur_gradw, facteur_gradb, gamma):
        """
        Train the network for a single sample.
        
        Args:
            input_data (numpy.ndarray): Input data
            target_output (numpy.ndarray): Target output
            facteur_gradw (dict): Gradient factors for weights
            facteur_gradb (dict): Gradient factors for biases
            gamma (float): Momentum factor
            
        Returns:
            tuple: (loss, updated gradient factors for weights, updated gradient factors for biases)
        """
        gradients, loss = self.backprop(input_data, target_output)

        for c in range(1, self.C):
            if self.method == adagrad or self.method == momentum: 
                self.parametres['W' + str(c)], facteur_gradw['W' + str(c)] = self.method(self.parametres['W' + str(c)], gradients['deltaw' + str(c)], self.alpha, facteur_gradw['W'+str(c)], gamma)   # Update weights
                self.parametres['b' + str(c)], facteur_gradb['b ' + str(c)] = self.method(self.parametres['b' + str(c)], gradients['deltab' + str(c)], self.alpha, facteur_gradb['b'+str(c)], gamma) 
            
            elif self.method == adam: 
                self.parametres['W' + str(c)], facteur_gradw['W' + str(c)], facteur_gradw['SW' + str(c)] = self.method(self.parametres['W' + str(c)], gradients['deltaw' + str(c)], self.alpha, facteur_gradw['W'+str(c)], facteur_gradw['SW'+ str(c)], gamma)   # Update weights
                self.parametres['b' + str(c)], facteur_gradb['b ' + str(c)], facteur_gradb['SB' + str(c)] = self.method(self.parametres['b' + str(c)], gradients['deltab' + str(c)], self.alpha, facteur_gradb['b'+str(c)], facteur_gradb['SB'+str(c)], gamma)
            
            else:
                self.parametres['W' + str(c)] = self.method(self.parametres['W' + str(c)], gradients['deltaw' + str(c)], self.alpha)   # Update weights
                self.parametres['b' + str(c)] = self.method(self.parametres['b' + str(c)], gradients['deltab' + str(c)], self.alpha) 
        return (loss, facteur_gradw, facteur_gradb)

    def train_set(self, epoch, gamma, input_data, target_output, test_input, test_output, error_evaluate):
        """
        Train the network on a dataset.
        
        Args:
            epoch (int): Number of training epochs
            gamma (float): Momentum factor
            input_data (numpy.ndarray): Training input data
            target_output (numpy.ndarray): Training target output
            test_input (numpy.ndarray): Test input data
            test_output (numpy.ndarray): Test target output
            error_evaluate (bool): Whether to evaluate error during training
            
        Returns:
            tuple or list: Error rates or loss list depending on error_evaluate
        """
        error_ratetrain = []
        error_ratetest = []
        loss_list = []
        n = np.shape(input_data)[0]
        facteur_gradw = {'W' + str(i): 0 for i in range(1, self.C)}
        facteur_gradb = {'b'+str(i): 0 for i in range(1, self.C)}
        if self.method == adam:
            for c in range(1, self.C):
                facteur_gradw['SW' + str(c)] = 0
                facteur_gradb['SB' + str(c)] = 0
        for j in range(1, epoch+1):
            error_avgtrain = 0
            error_avgtest = 0
            for i in range(n):
                if error_evaluate:
                    loss, facteur_gradw, facteur_gradb = self.train(input_data[i], target_output[i], facteur_gradw, facteur_gradb, gamma)
                    p = self.predict(input_data[i])
                    if (np.argmax(target_output[i]) >= 1 and np.argmax(p) == 0) or (np.argmax(target_output[i]) == 0 and np.argmax(p) >= 1): 
                        error_avgtrain += 1

                else:  
                    loss, facteur_gradw, facteur_gradb = self.train(input_data[i], target_output[i], facteur_gradw, facteur_gradb, gamma)
                    loss_list.append(loss)
            if j % 5 == 0 and error_evaluate:
                n_test = np.shape(test_input)[0]
                for k in range(n_test):
                    p = self.predict(test_input[k])
                    if (np.argmax(test_output[k]) >= 1 and np.argmax(p) == 0) or (np.argmax(test_output[k]) == 0 and np.argmax(p) >= 1):  
                        error_avgtest += 1
                error_ratetest.append(error_avgtest/n_test)
                print('Test error: ' + str(error_avgtest/n_test))
            error_ratetrain.append(error_avgtrain/n)
 
            print(str(j*100/epoch) + "%", "Training error: " + str(error_avgtrain/n))
        
        if error_evaluate: 
            return(error_ratetrain, error_ratetest)

        return(loss_list)

    def predict(self, input_data):
        """
        Make a prediction for the given input.
        
        Args:
            input_data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted output
        """
        activations, dico_Z = self.feedforward(input_data)
        
        return(activations['A'+str(self.C-1)])

    def lr_determine(self, gamma, input_data, target_output):
        """
        Determine the optimal learning rate.
        
        Args:
            gamma (float): Momentum factor
            input_data (numpy.ndarray): Input data
            target_output (numpy.ndarray): Target output
        """
        parametres_init = self.parametres
        n = np.shape(input_data)[0]
        X = []
        Y = []
        for i in range(100):
            self.alpha = (10**(-5))*exp(log(100/(10**(-5)))*i/100)
            loss_list = self.train_set(1, gamma, input_data, target_output, 0, 0, error_evaluate=False)
            loss_avg = (1/n)*sum(loss_list)
            print('Learning rate determination...' + str(i))
            X.append(self.alpha)
            Y.append(loss_avg)
            self.parametres = parametres_init
        np.save('lr_determine', np.array(X))
        np.save('loss_determine', np.array(Y))
        plt.plot(X, Y)
        plt.xscale('log')
        plt.show()

    def save_model(self, model_name):
        """
        Save the model to a file.
        
        Args:
            model_name (str): Name of the model file
        """
        return(pickle.dump(self, open(model_name, "wb")))
    
    @staticmethod
    def load_model(model_name):
        """
        Load a model from a file.
        
        Args:
            model_name (str): Name of the model file
            
        Returns:
            NeuralNetwork: Loaded model
        """
        return(pickle.load(open(model_name, "rb")))