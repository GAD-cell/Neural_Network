from math import*
from random import*
import matplotlib.pyplot as plt
import numpy as np
from Fonctions import*
import pickle

class NeuralNetwork():

    def __init__(self, Reseau ,  fonction_activation , fonction_cout , fonction_gradient , learning_rate ):
        self.act = fonction_activation  
        self.cost = fonction_cout             
        self.method = fonction_gradient 
        self.alpha = learning_rate
        self.parametres = {}
        self.C=len(Reseau)
        for c in range(1,self.C):
            n = Reseau[c]
            m = Reseau[c-1]
            self.parametres['W'+ str(c)] = np.random.uniform(low=-sqrt(6)/sqrt(n+m), high=sqrt(6)/sqrt(n+m), size=(n, m) )
            self.parametres['b'+str(c)] = np.zeros((Reseau[c], 1))

    def feedforward(self,x):
        activations = {'A0' : x }
        dico_Z = {}
        
        for c in range(1,self.C):
            Z = np.dot(self.parametres['W' + str(c)],activations['A' + str(c-1)]) + self.parametres['b' + str(c)]
            dico_Z['Z' + str(c)] = Z
            activations['A' + str(c)] = self.act(Z)[0]
            
        return( activations, dico_Z )

    def backprop(self, entree , sortie):
        gradients = {}
        activations, dico_Z = self.feedforward(entree)
        deltaZ = self.cost(activations['A' + str(self.C-1)], sortie)[1]
        loss = self.cost(activations['A' + str(self.C-1)], sortie)[0]
        for c in reversed(range(1, self.C)):
            gradients['deltaw' + str(c)] = np.dot(np.multiply(deltaZ,self.act(dico_Z['Z' + str(c)])[1]) , activations['A'+str(c-1)].T)          #DL/DW
            gradients['deltab' + str(c)] = np.multiply(deltaZ,self.act(dico_Z['Z' + str(c)])[1])                                                #DL/DB
            
            deltaZ = np.dot(self.parametres['W' + str(c)].T , np.multiply(deltaZ , self.act(dico_Z['Z' + str(c)])[1]))                          #DL/DX qu'on calcule par rétropropagation
        
        return(gradients , loss)
 
    def train(self, entree , sortie , facteur_gradw , facteur_gradb , gamma): #facteur_grad sert pour les plusieurs methodes pour actualiser les parametres
        gradients , loss = self.backprop(entree , sortie)

        for c in range(1 , self.C ):
            if self.method == adagrad or self.method == momentum : 
                self.parametres['W' + str(c)] , facteur_gradw['W' + str(c)] = self.method(self.parametres['W' + str(c)] , gradients['deltaw' + str(c) ], self.alpha , facteur_gradw['W'+str(c)], gamma)   #actualisation des W
                self.parametres['b' + str(c)] , facteur_gradb['b ' + str(c)] = self.method(self.parametres['b' + str(c)] , gradients['deltab' + str(c) ], self.alpha , facteur_gradb['b'+str(c)] , gamma) 
            
            elif self.method == adam : 
                self.parametres['W' + str(c)] , facteur_gradw['W' + str(c)] , facteur_gradw['SW' + str(c)] = self.method(self.parametres['W' + str(c)] , gradients['deltaw' + str(c) ], self.alpha , facteur_gradw['W'+str(c)],facteur_gradw['SW'+ str(c)],  gamma)   #actualisation des W
                self.parametres['b' + str(c)] , facteur_gradb['b ' + str(c)] , facteur_gradb['SB' + str(c)] = self.method(self.parametres['b' + str(c)] , gradients['deltab' + str(c) ], self.alpha , facteur_gradb['b'+str(c)], facteur_gradb['SB'+str(c)] , gamma)
            
            else :
                self.parametres['W' + str(c)] = self.method(self.parametres['W' + str(c)] , gradients['deltaw' + str(c) ], self.alpha )   #actualisation des W
                self.parametres['b' + str(c)] = self.method(self.parametres['b' + str(c)] , gradients['deltab' + str(c) ], self.alpha ) 
        return (loss, facteur_gradw , facteur_gradb)

    def train_set(self , epoch , gamma , entree , sortie , entree_test, sortie_test , error_evaluate):
        error_ratetrain = []
        error_ratetest = []
        loss_list = []
        n = np.shape(entree)[0]
        facteur_gradw = {'W' + str(i) : 0 for i in range(1 , self.C) }
        facteur_gradb = {'b'+str(i) : 0 for i in range(1,self.C)}
        if self.method == adam :
            for c in range(1 , self.C):
                facteur_gradw['SW' + str(c)] = 0
                facteur_gradb['SB' + str(c)] = 0
        for j in range(1 , epoch+1):
            error_avgtrain = 0
            error_avgtest =  0
            for i in range(n):
                if error_evaluate :
                    loss , facteur_gradw , facteur_gradb = self.train(entree[i], sortie[i] ,facteur_gradw , facteur_gradb , gamma)
                    p = self.predict(entree[i])
                    if (np.argmax(sortie[i]) >=1 and np.argmax(p) == 0) or (np.argmax(sortie[i]) == 0 and np.argmax(p) >= 1) : 
                        error_avgtrain += 1

                else :  
                    loss , facteur_gradw , facteur_gradb = self.train(entree[i], sortie[i] ,facteur_gradw , facteur_gradb, gamma)
                    loss_list.append(loss)
            if j%5==0 and error_evaluate :
                n_test = np.shape(entree_test)[0]
                for k in range(n_test):
                    p = self.predict(entree_test[k])
                    if (np.argmax(sortie_test[k]) >= 1 and np.argmax(p) == 0) or (np.argmax(sortie_test[k]) == 0 and np.argmax(p) >= 1) :  
                        error_avgtest +=1
                error_ratetest.append(error_avgtest/n_test)
                print('erreur_test: ' +str(error_avgtest/n_test))
            error_ratetrain.append(error_avgtrain/n)
 
            print(str(j*100/epoch)+"%" , "erreur_train : " + str(error_avgtrain/n))
        
        if error_evaluate : 
            return(error_ratetrain , error_ratetest)

        return(loss_list)

    def predict(self, entree):

        activations, dico_Z = self.feedforward(entree)
        
        return(activations['A'+str(self.C-1)])


    def lr_determine(self , gamma , entree , sortie ):
        parametres_init = self.parametres
        n = np.shape(entree)[0]
        X = []
        Y = []
        for i in range(100):
            self.alpha = (10**(-5))*exp(log(100/(10**(-5)))*i/100)
            loss_list = self.train_set(1 ,gamma, entree , sortie , 0 , 0 ,  error_evaluate = False )
            loss_avg = (1/n)*sum(loss_list)
            print('lr.determine...'+ str(i))
            X.append(self.alpha)
            Y.append(loss_avg)
            self.parametres = parametres_init
        np.save('lr_determine' , np.array(X) )
        np.save('loss_determine', np.array(Y) )
        plt.plot(X,Y)
        plt.xscale('log')
        plt.show()


    def save_model(self , nom_modele):
        return(pickle.dump(self , open(nom_modele,"wb")))
    
    def load_model(nom_modele):
        return(pickle.load(open(nom_modele,"rb")))


