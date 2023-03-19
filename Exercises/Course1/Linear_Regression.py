import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([500.0, 800.00])
print(f"x_train = {x_train}")
print(f"x_train = {y_train}")

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i},{y_i})")

plt.scatter(x_train, y_train)
plt.title("Linear Model Demo")
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.show()

w = 100
b = 100
print(f"W = {w}, B = {100}")

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, y_train, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker = 'x', c = 'r', label = 'Actual Values')
plt.title("machine Learning Model")
plt.ylabel("Y")
plt.xlabel("X")
plt.legend()
plt.show()