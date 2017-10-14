import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
#model.add(Dense(10,input_dim=9,activation='tanh'))
model.add(Dense(10,input_shape=[9,],activation='tanh'))
print model.predict(np.ones((3,3)).reshape(1,9))


