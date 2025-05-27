from math import*
from random import*
import numpy as np

"""
This module contains activation functions, cost functions, and gradient descent optimization methods
used by the neural network implementation.
"""

# Activation Functions

def sigmoid(X):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    
    Args:
        X (numpy.ndarray): Input values
        
    Returns:
        tuple: (function values, derivatives)
    """
    fonction = 1/(1+np.exp(-X))
    derive = np.multiply(fonction, 1-fonction)
    return(fonction, derive)

def ReLU(X):
    """
    Rectified Linear Unit activation function: f(x) = max(0, x)
    
    Args:
        X (numpy.ndarray): Input values
        
    Returns:
        tuple: (function values, derivatives)
    """
    a = np.zeros(X.shape, dtype=np.float32)
    fonction = np.maximum(a, X)
    derive = np.greater(X, 0.).astype(np.float32)
    return(fonction, derive) 

def tanh(X):
    """
    Hyperbolic tangent activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        X (numpy.ndarray): Input values
        
    Returns:
        tuple: (function values, derivatives)
    """
    fonction = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    derive = 1 - fonction**2
    return(fonction, derive) 

 
# Cost Functions

def meansquare(x, sortie):
    """
    Mean squared error cost function
    
    Args:
        x (numpy.ndarray): Predicted values
        sortie (numpy.ndarray): Target values
        
    Returns:
        tuple: (cost value, derivative)
    """
    n = np.shape(sortie)[0]
    fonction = (1/n)*np.sum(np.multiply(x-sortie, x-sortie))
    derive = (2/n)*(x-sortie)
    return(fonction, derive)

def logloss(X, sortie):
    """
    Binary cross-entropy (log loss) cost function
    
    Args:
        X (numpy.ndarray): Predicted values
        sortie (numpy.ndarray): Target values
        
    Returns:
        tuple: (cost value, derivative)
    """
    n = np.shape(sortie)[0]
    fonction = (-1)*np.sum(sortie*np.log(X) + (1-sortie)*np.log(1-X))/n
    derive = (-1)*((sortie/X) - (1-sortie)/(1-X))/n
    return(fonction, derive)

# Gradient Descent Optimization Methods

def sgd(x, deltax, alpha):
    """
    Stochastic Gradient Descent optimization
    
    Args:
        x (numpy.ndarray): Current parameter values
        deltax (numpy.ndarray): Gradient
        alpha (float): Learning rate
        
    Returns:
        numpy.ndarray: Updated parameter values
    """
    return(x - alpha*deltax)

def momentum(x, deltax, alpha, velocity, gamma):
    """
    Momentum optimization
    
    Args:
        x (numpy.ndarray): Current parameter values
        deltax (numpy.ndarray): Gradient
        alpha (float): Learning rate
        velocity (numpy.ndarray): Current velocity
        gamma (float): Momentum factor
        
    Returns:
        tuple: (updated parameter values, updated velocity)
    """
    velocity = gamma*velocity + alpha*deltax
    return(x-velocity, velocity)

def adagrad(x, deltax, alpha, eta, epsilon):
    """
    Adagrad optimization
    
    Args:
        x (numpy.ndarray): Current parameter values
        deltax (numpy.ndarray): Gradient
        alpha (float): Learning rate
        eta (numpy.ndarray): Accumulated squared gradients
        epsilon (float): Small constant for numerical stability
        
    Returns:
        tuple: (updated parameter values, updated accumulated squared gradients)
    """
    eta += np.multiply(deltax, deltax)
    return(x-alpha*deltax/(np.sqrt(eta+epsilon)), eta) 

def adam(x, deltax, alpha, velocity, s, epsilon):
    """
    Adam optimization
    
    Args:
        x (numpy.ndarray): Current parameter values
        deltax (numpy.ndarray): Gradient
        alpha (float): Learning rate
        velocity (numpy.ndarray): First moment estimate
        s (numpy.ndarray): Second moment estimate
        epsilon (float): Small constant for numerical stability
        
    Returns:
        tuple: (updated parameter values, updated first moment, updated second moment)
    """
    velocity = 0.5*velocity + (1-0.5)*deltax
    s = 0.99*s + (1-0.99)*np.multiply(deltax, deltax)
    sprime = s/0.01
    velocityprime = velocity/0.5
    return(x-alpha*velocityprime/(np.sqrt(sprime) + epsilon), velocity, s)