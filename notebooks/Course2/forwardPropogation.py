import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from keras.models import Model


x = np.array([[200.0, 17.0, 60.0, 120.0]]) 
layer_1 = tf.keras.layers.Dense(units = 4, activation = "sigmoid", name = "layer_1")
layer_2 = tf.keras.layers.Dense(units = 1, activation = "sigmoid", name = "layer_2")
a1 = layer_1(x)
a2 = layer_2(a1) #output function
#model = Sequential([
 #   Input(shape = (4,2)),
  #  tf.keras.layers.Dense(units = 3, activation = "sigmoid", name = "layer_1"), 
  #  tf.keras.layers.Dense(units = 1, activation = "sigmoid", name = "layer_2")
  #  ])
print(a2)
#model.predict(x)
#model.summary()




