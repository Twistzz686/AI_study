import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

x = np.linspace(-10,10,100)

y_tanh = tanh(x)
y_derivative = tanh_derivative(x)



def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)

x = np.linspace(-10,10,100)
y_relu = relu(x)
y_derivative = relu_derivative(x)



def softmax(x):
    vals = np.exp(x)
    return vals / np.sum(vals)

def softmax_derivative(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s,s)

x = np.linspace(-5,5,100)
y_softmax = softmax(x)
y_derivative = softmax_derivative(x)



def relu(x):
    return np.maximum(0,x)

def leaky_relu(x,alpha = 0.01):
    return np.where(x>0,x,alpha * x)

def leaky_derivative(x,alpha = 0.01):
    return np.where(x>0,1,alpha)


x = np.linspace(-3,3,100)





def relu(x):
    return np.maximum(0,x)

def leaky_relu(x,alpha = 0.25):
    return np.where(x>0,x,alpha * x)

def leaky_derivative(x,alpha = 0.01):
    return np.where(x>0,1,alpha)


x = np.linspace(-3,3,100)



def relu(x):
    return np.maximum(0,x)

def elu(x,alpha = 0.25):
    return np.where(x>0,x,alpha * (np.exp(x) - 1))

def elu_derivative(x,alpha = 0.25):
    return np.where(x>0,1,alpha * np.exp(x))