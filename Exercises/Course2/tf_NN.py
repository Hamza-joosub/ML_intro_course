import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#data
X = np.array([[69527371, 30.7075],[76140310, 22.2051],[7549, 51.33],[84854656, 146.6651],[1331, 109.78],[50141098,23.9036],[98944633, 26.429]]) 
Y = np.array([[0.011694],[0.013753],[-0.027954],[0.007204], [0.008152],[-0.045479],[-0.005454]])
print(f"x data shape = {X.shape} and y shape = {Y.shape}")

#normalization
print(f"pre-norm-max/min-temp {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}\n")
print(f"pre-Norm-max/min-time: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}\n")
norm_l = tf.keras.layers.Normalization()
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"post-norm-max/min-Volume {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"post-Norm-max/min-P/E: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

#copy data 

Xt = np.tile(Xn,(10000,1))
Yt= np.tile(Y,(10000,1))   
#print(Xt.shape, Yt.shape) 


#create Model
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(4, activation='relu', name = 'layer1'),
        Dense(4, activation = 'relu', name = 'layer2'),
        Dense(1, activation='linear', name = 'output')
     ]
)
model.summary()

# get instantiated weights and biases
W1,b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
#print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
#print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

#compile model
model.compile(
    loss = "mean_squared_error", 
    optimizer =  tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['mean_squared_error']
    )


#Back Propogation
model.fit(Xt,Yt,epochs=30)

#get updated weights and biases
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
#print("W1:\n", W1, "\nb1:", b1)
#print("W2:\n", W2, "\nb2:", b2)

#Forward Propogation
X_test = np.array([[87300242,97.0362]])  # postive example y = -0,010896])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)