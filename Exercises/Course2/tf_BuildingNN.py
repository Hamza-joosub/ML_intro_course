import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model

layer_1 = Dense(units = 3, activation = "sigmoid", name = "layer_1")
layer_2 = Dense(units = 1, activation = "sigmoid", name = "layer_2")

x_train = np.array([[200,17],[120,5],[425,20],[212,18]])
y_train = np.array([1,0,0,1])

x_test = np.array([210,13])

model = Sequential([
    Dense(units = 3, activation = "sigmoid", name = "layer_1"), 
    Dense(units = 1, activation = "sigmoid", name = "layer_2")
                        ])
model.compile()
model.fit(x_train, y_train)
model.summary()
model.predict(x_test) #forward propogation

