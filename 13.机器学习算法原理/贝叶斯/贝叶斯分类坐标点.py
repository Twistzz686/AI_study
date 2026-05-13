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

X = np.concatenate((points1,points2))
y = np.concatenate((np.zeros(len(points1)),np.ones(len(points2))))

prior_probabilities = [np.sum(y == 0) / len(y),np.sum(y == 1) / len(y)]

class_means = [np.mean(X[y == 0],axis = 0),np.mean(X[y == 1],axis = 0)]

X_y_0 = X[y == 0].T
X_y_1 = X[y == 1].T
class_covs = [np.cov(X_y_0),np.cov(X_y_1)]

point = np.array([0.5,3])

def pdf(x,mean,cov):
    n = len(mean)
    coeff = 1 / (2 * np.pi) ** (n/2) * np.sqrt(np.linalg.det(cov))
    exponet = -0.5 * np.dot(np.dot((x - mean).T,np.linalg.inv(cov)),(x - mean))
    return coeff * np.exp(exponet)

posterior_probabilities = []
for i in range(2):
    likelihood = pdf(point,class_means[i],class_covs[i])
    posterior_probabilities.append(prior_probabilities[i] * likelihood)


pre_class = np.argmax(posterior_probabilities)

print(f"点{point}属于类别{pre_class}")