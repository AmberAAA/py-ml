from lib.sigmoid import sigmoid
from lib.load import load_mat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def init_dataset():
    data = load_mat('./ex3data1.mat')
    return data['X'], data['y']


def draw_one(data):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)


def cost_function(theta, X, y, lamda):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    J = 0
    J = J + np.sum(-y.T.dot((np.log(h))) - (1 - y).T.dot(np.log(1 - h))) / m + np.sum(np.power(theta[1:, 0], 2)) * lamda / 2 / m
    gradient = X.T.dot(h - y) / m + lamda / m * theta
    gradient[0] = (X.T.dot(h - y) / m)[0]
    # print(gradient)
    return J


def gradient_d(theta, X, y, lamda):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m + lamda / m * theta
    gradient[0] = (X.T.dot(h - y) / m)[0]
    return gradient


if __name__ == '__main__':
    X, y = init_dataset()
    (m, n) = X.shape

    # 先跳选100个绘制一下
    select = np.random.permutation(m)[0: 100]
    selectX = X[select, :]
    for i in range(100):
        plt.subplot(10, 10, i+1)
        data = selectX[i, :].reshape((20, 20)).T
        draw_one(data)
    plt.show()

    # add one to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # init theta
    init_theta = np.zeros((n+1, 1))

    # test cost function
    theta_t = np.array([-2, - 1, 1, 2]).reshape((4, 1))
    X_t = np.concatenate((np.ones((5, 1)), np.arange(1, 16).reshape((5, 3), order='F') / 10), axis=1)
    y_t = np.array([1, 0, 1, 0, 1]).reshape((5, 1))
    lambda_t = 3
    out = cost_function(theta_t, X_t, y_t, lambda_t)









