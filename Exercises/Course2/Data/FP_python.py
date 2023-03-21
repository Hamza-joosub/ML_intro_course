import numpy as np

X = np.array([[3,6],[4,2]])

#singe neuron of layer 1
a1 = np.zeros(2,)
W1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(W1_1,X)+b1_1
a1_1 = sigmoid(z1_1)
a1.append(a1_1)

def Dense(a_in,W,b):
    units  = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:j]
        z = np.dot(w,a_in) + b[j]
        a_out[j] = g(z)
    return a_out
    
def Sequential():
    a1 = Dense(X,W1,b1)
    a2 = Dense(a1,W2,b2)
    topG = a2
    return topG

#single neuron for array 1
a2 = np.zeros
W2_1 = np.array([[3,5]])
b2_1 = np.array([3])
z2_1 = np.dot(W2_1,a1)+ b2_1
