import numpy as np 
import matplotlib.pyplot as plt
from random import*
from sklearn.utils import shuffle
import pickle



error_train = []
error_test = []

'''
for j in range(400):
    error_avgtrain = 0
    error_avgtest = 0
    
    for i in range(0,400):
        a = nn.train(X_train[i],y_train[i])
        p = nn.predict(X_train[i])
        if (np.argmax(y_train[i]) >=1 and np.argmax(p) == 0) or (np.argmax(y_train[i]) ==0 and np.argmax(p) >= 1) :
            error_avgtrain +=1
    error_train.append(error_avgtrain/400)
    if j % 10 == 0:
        for i in range(100):
            p = nn.predict(X_test[i])
            if (np.argmax(y_test[i]) >=1 and np.argmax(p) == 0) or (np.argmax(y_test[i]) ==0 and np.argmax(p) >= 1) :
                error_avgtest +=1
        error_test.append(error_avgtest/100)
        print("erreur_test : " +str(error_avgtest/100))
    
    print(str(j*100/500)+"%" , "erreur_train : " + str(error_avgtrain/400))
'''
'''Enregistrement du modèle'''
'''
nn.save_model("NN1")

pickle_train = open("train_error1" , "wb")
pickle_test = open("test_error1" , "wb")
pickle.dump(error_train , pickle_train)
pickle.dump(error_test , pickle_test)
pickle_train.close()
pickle_test.close()
'''

loss_r_m_s = np.load(r'C:\Users\Sinoué\Documents\TIPE\valeur_courbes\loss_determine_r_m_s.npy')
loss_s_m_s = np.load(r'C:\Users\Sinoué\Documents\TIPE\valeur_courbes\loss_determine_s_m_s.npy')
loss_t_m_s = np.load(r'C:\Users\Sinoué\Documents\TIPE\valeur_courbes\loss_determine_t_m_s.npy')
lr = np.load(r'C:\Users\Sinoué\Documents\TIPE\valeur_courbes\lr_determine_r_m_s.npy')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(lr, loss_r_m_s ,'r')
ax1.plot(lr, loss_s_m_s , 'g')
ax1.set_xlabel('learning_rate')
ax1.set_ylabel('loss')


ax2.plot(lr, loss_t_m_s , 'b')
ax2.set_ylabel('loss',color = 'blue')
ax2.tick_params(axis='y', labelcolor='blue')
  

plt.xscale('log')
fig.tight_layout()


    