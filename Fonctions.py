from math import*
from random import*
import numpy as np

'''Fonctions_activations'''

def sigmoid(X):
    fonction = 1/(1+np.exp(-X))
    derive = np.multiply(fonction,1-fonction)
    return(fonction , derive)

def ReLU(X):
    a = np.zeros(X.shape,dtype=np.float32)
    fonction = np.maximum(a , X )
    derive = np.greater(X , 0.).astype(np.float32)
    return(fonction , derive) 

def tanh(X):
    fonction = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    derive = 1 - fonction**2
    return(fonction , derive) 

 
'''Fonctions_coût'''

def meansquare(x , sortie):
    n = np.shape(sortie)[0]
    fonction = (1/n)*np.sum(np.multiply(x-sortie,x-sortie))
    derive = (2/n)*(x-sortie)
    return(fonction , derive)

def logloss(X , sortie):
    n = np.shape(sortie)[0]
    fonction = (-1)*np.sum(sortie*np.log(X) + (1-sortie)*np.log(1-X))/n
    derive = (-1)*((sortie/X) - (1-sortie)/(1-X) )/n
    return(fonction , derive)

'''Fonctions_gradient'''

def sgd( x , deltax , alpha ):
    return(x - alpha*deltax)

def momentum(x , deltax , alpha , velocity, gamma ):
    velocity =  gamma*velocity + alpha*deltax
    return(x-velocity, velocity)

def adagrad(x , deltax , alpha , eta , epsilon):
    eta += np.multiply(deltax,deltax)
    return(x-alpha*deltax/(np.sqrt(eta+epsilon)), eta) 

def adam (x , deltax , alpha , velocity , s , epsilon):
    velocity =  0.5*velocity + (1-0.5)*deltax
    s = 0.99*s + (1-0.99)*np.multiply(deltax ,deltax )
    sprime = s/0.01
    velocityprime = velocity/0.5
    return(x-alpha*velocityprime/(np.sqrt(sprime) + epsilon) , velocity , s)









