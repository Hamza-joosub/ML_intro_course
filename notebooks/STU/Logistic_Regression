import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0, 1, 2],[ 3, 4, 5],[5, 8, 1], [11, 5, 3]) #add data 
y_train = np.array([0, 0, 0, 1, 1, 1])


#sets data to class catagories
pos = y_train == 1
neg = y_train == 0

def sigmoid_function(z):
    return 1/(1+np.exp(-z))

def compute_cost(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    non_regularized_cost = 0.0
    
    for i in range(m):
        z_i = np.dot(X[i],w[i])+b
        f_wb_i = sigmoid_function(z_i)
        non_regularized_cost = non_regularized_cost + -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    non_regularized_cost = non_regularized_cost/m
    
    regularized_cost = 0.0
    for j in range(n):
        regularized_cost =  regularized_cost + w[j]**2
    regularized_cost = (lambda_/(2*m)) * regularized_cost
    regularized_cost = regularized_cost + non_regularized_cost
    return regularized_cost

def compute_gradient(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    dj_dw = np.zeros(X.shape)
    dj_db = 0
    
    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid_function(z_i)
        error_i = f_wb_i  - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error_i * X[i,j]
        dj_db = dj_db + error_i
        
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
        
        return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)

    J_History = []
    w_history = []

    for i in range[num_iters]:
        dj_db, dj_dw = compute_gradient(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_History.append(cost)
    return w_in, b_in, J_History, w_history

def prediction(X, w, b):
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):   
        f_wb = sigmoid_function(np.dot(X[i],w) + b)

        # Apply the threshold
        p[i] = 1.0 if (f_wb >= 0.5) else 0.0
    
    return p

    

    
    


