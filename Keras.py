from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten
import numpy as np

model = Sequential()
model.add(Conv2D(32 , (3,3) , activation = 'relu' , input_shape = (100,100,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32 , (3,3) , activation = 'relu' ))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
print(model.layers[4].output)