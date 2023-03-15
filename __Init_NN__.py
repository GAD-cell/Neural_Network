from Neural_Network import*
import numpy as np
from Fonctions import*

'''Initialisation du réseau'''
Reseau = [100*100 , 50 , 30 , 5]
nn = NeuralNetwork(Reseau , sigmoid , meansquare , adam , 0.05) 

'''Implémentation des données d'entrainements'''
X_train, y_train = (np.load(r"C:\Users\Sinoué\Documents\TIPE\Neural_Network\train_image100pdroite.npy")/255).astype(float), np.load(r"C:\Users\Sinoué\Documents\TIPE\Neural_Network\train_labels.npy").astype(float)
X_train = np.reshape(X_train , (400,100*100,1))

X_test , y_test = (np.load(r"C:\Users\Sinoué\Documents\TIPE\Neural_Network\test_image100pdroite.npy")/255).astype(float), np.load(r"C:\Users\Sinoué\Documents\TIPE\Neural_Network\test_labels.npy").astype(float)
X_test = np.reshape(X_test , (100,100*100,1))

'''Entrainement du réseau'''

error_train, error_test = nn.train_set(400 , 10**(-8) ,  X_train , y_train , X_test , y_test , error_evaluate = True)
#nn.save_model('NN2')
np.save('error_train' , error_train)
np.save('error _test' , error_test)

'''Determination learning_rate'''

#nn.lr_determine(10**(-8) , X_train  , y_train)
