import numpy as np


points = np.array([[0.8,0],
                   [1.1,0],
                   [1.7,0],
                   [1.9,0],
                   [2.7,1],
                   [3.2,1],
                   [3.7,1],
                   [4.0,1],
                   [5.0,0],
                   [5.5,0],
                   [6.0,0],
                   [6.3,0]])


X = points[:,0]
Y = points[:,1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(w11_1,b1_1,w12_1,b2_1,w11_2,b1_2,w21_2,b2_2):
    z1_1 = w11_1 * X + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1 * X + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_1,a2_1,a1_2

w11_1,b1_1,w12_1,b2_1,w11_2,w21_2,b1_2 = 0.1,0.6,0.9,0,-1.5,0.1,0.9
lr = 0.5

def loss_func(Y,y_hat):
    loss = np.mean((Y-y_hat) ** 2)
    return loss

epoches = 5000
for epoch in range(1,epoches + 1):
    a1_1,a2_1,a1_2 = forward(w11_1,b1_1,w12_1,b2_1,w11_2,w21_2,b1_2,X)
    loss = loss_func(Y,a1_2)


    deda1_2 = -2 * (Y - a1_2)
    dedz1_2 = da1_2dz1_2 = deda1_2 * a1_2 * (1 - a1_2)

    dedw11_2 = np.mean(dedz1_2 * a1_1)
    dedw21_2 = np.mean(dedz1_2 * a2_1)
    dedb1_2 = np.mean(dedz1_2)

    deda1_1 = dedz1_2 * w11_2
    dedz1_1 = deda1_1 * a1_1 * (1 - a1_1)
    dedw11_1 = np.mean(dedz1_1 * X)
    dedb1_1 = np.mean(dedz1_1)

    deda2_1 = dedz1_2 * w21_2
    dedz2_1 = deda2_1 * a2_1 * (1 - a2_1)
    dedw12_1 = np.mean(dedz2_1 * X)
    dedb2_1 = np.mean(dedz2_1)

    w11_2 -= lr * dedw11_2
    w21_2 -= lr * dedw21_2
    b1_2 -= lr * dedb1_2

    w11_1 -= lr * dedw11_1
    b1_1 -= lr * dedb1_1
    w12_1 -= lr * dedw12_1
    b2_1 -= lr * dedb2_1

    if epoch == 1 or epoch % 100 == 0:
        print(f"epoch:{epoch},loss:{loss}")