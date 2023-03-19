import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


#data
X = np.array([[1,7],[28,3],[1,5],[7,13],[8,40],[13,8],[18,6],[2,14],[3,11],[4,21]]) 
Y = np.array([[1],[0],[0],[0], [1],[0],[1],[0],[1], [0]])
print(f"x data shape = {X.shape} and y shape = {Y.shape}")

#normalization
print(f"pre-norm-max/min-temp {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"pre-Norm-max/min-time: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"post-norm-max/min-temp {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"post-Norm-max/min-time: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

#copy data 
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape) 

#create Model
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
model.summary()

# get instantiated weights and biases
W1,b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

#compile model
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

#fit model
model.fit(
    Xt,Yt,            
    epochs=10,
)

#get updated weights and biases
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

#inference
X_test = np.array([
    [188,11],  # postive example
    [210,15]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)