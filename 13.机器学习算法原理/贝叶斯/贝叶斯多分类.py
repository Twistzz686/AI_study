import numpy as np
import matplotlib.pyplot as plt




points1 = np.array([[1.9,1.2],
                    [1.5,2.1],
                    [1.9,0.5],
                    [1.5,0.9],
                    [0.9,1.2],
                    [1.1,1.7],
                    [1.4,1.1]])
points2 = np.array([[3.2,3.2],
                    [3.7,2.9],
                    [3.2,2.6],
                    [1.7,3.3],
                    [3.4,2.6],
                    [4.1,2.3],
                    [3.0,2.9]])
points3 = np.array([[3.3,1.2],
                    [3.8,0.9],
                    [3.3,0.6],
                    [2.8,1.3],
                    [3.5,0.6],
                    [4.2,0.3],
                    [3.2,0.9]])

X = np.concatenate((points1,points2,points3))
y = np.concatenate((np.zeros(len(points1)),np.ones(len(points2)),np.ones(len(points3)) + 1))

prior_probabilities = [np.sum(y == 0) / len(y),np.sum(y == 1) / len(y),np.sum(y == 2) / len(y)]

class_means = [np.mean(X[y == 0],axis = 0),np.mean(X[y == 1],axis = 0),np.mean(X[y == 2],axis = 0)]

X_y_0 = X[y == 0].T
X_y_1 = X[y == 1].T
X_y_2 = X[y == 2].T
class_covs = [np.cov(X_y_0),np.cov(X_y_1),np.cov(X_y_2)]

xx,yy = np.meshgrid(np.arange(0,5,0.05),np.arange(0,4,0.05))

grid_points = np.np.c_[xx.ravel(),yy.ravel()]


def pdf(x,mean,cov):
    n = len(mean)
    coeff = 1 / (2 * np.pi) ** (n/2) * np.sqrt(np.linalg.det(cov))
    exponet = -0.5 * np.dot(np.dot((x - mean).T,np.linalg.inv(cov)),(x - mean))
    return coeff * np.exp(exponet)

posterior_probabilities = []
for i in range(3):
    likelihood = pdf(point,class_means[i],class_covs[i])
    posterior_probabilities.append(prior_probabilities[i] * likelihood)


pre_class = np.argmax(posterior_probabilities)

print(f"点{point}属于类别{pre_class}")