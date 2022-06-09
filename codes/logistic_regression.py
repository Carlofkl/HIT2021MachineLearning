'''
逻辑回归

目的：理解逻辑回归模型，掌握逻辑回归模型的参数估计算法。

要求：实现两种损失函数的参数估计（1，无惩罚项；2.加入对参数的惩罚），可以采用梯度下降、共轭梯度或者牛顿法等。

验证：1.可以手工生成两个分别类别数据（可以用高斯分布），验证你的算法。考察类条件分布不满足朴素贝叶斯假设，会得到什么样的结果。
2. 逻辑回归有广泛的用处，例如广告预测。可以到UCI网站上，找一实际数据加以测试。
'''

import math

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(wx):
    wx = wx.T
    size = wx.shape[0]
    for i in range(size):
        wx[i] = 1 / (1 + np.exp(-wx[i]))
    return wx

def generate_data(train_sample, navie = True):
    pos_mean = [0.8, 1] #正例的两维度均值
    neg_mean = [-0.8, -1] #反例的两维度均值
    X = np.zeros((2 * train_sample, 2))
    Y = np.zeros((2 * train_sample, 1))
    if navie:
        cov = np.mat([[0.3, 0], [0, 0.4]])
        X[:train_sample, :] = np.random.multivariate_normal(pos_mean, cov, train_sample)
        X[train_sample:, :] = np.random.multivariate_normal(neg_mean, cov, train_sample)
        Y[:train_sample] = 1
        Y[train_sample:] = 0
    else:
        cov = np.mat([[0.3, 0.5], [0.5, 0.4]])
        X[:train_sample, :] = np.random.multivariate_normal(pos_mean, cov, train_sample)
        X[train_sample:, :] = np.random.multivariate_normal(neg_mean, cov, train_sample)
        Y[:train_sample] = 1
        Y[train_sample:] = 0
    # print(X)
    # print(Y)
    # plt.plot(X[:train_sample, :], 'go')
    # plt.plot(X[train_sample:, :], 'ro')
    plt.scatter(X[:train_sample, 0], X[:train_sample, 1], c = 'b', marker = '.')
    plt.scatter(X[train_sample:, 0], X[train_sample:, 1], c = 'r', marker = '.')
    return X.T, Y

#损失函数，(不)带正则项(极大似然）,并对loss做归一化处理
def loss(X, Y, w, lamuda=0):
    size = X.shape[1]
    wX = np.zeros((size, 1))
    wX = np.dot(w, X).T
    part1 = np.dot(Y.T, wX)
    part2 = 0
    for i in range(size):
        part2 += np.log(1+np.exp(wX[i]))
    L_w = part1 - part2 - lamuda * np.dot(w, w.T) / 2
    return -L_w /size

#梯度下降法
def gradient_descent(X, Y, w, alpha=0.1, epsilon=0.1, lamuda = 0):
    cnt = 0 #记录迭代次数
    size = X.shape[1] #size为数据量
    new_loss = loss(X, Y, w, lamuda) #计算损失函数的值，通过比较损失函数的差结束迭代
    # print(new_loss)
    while True:
        cnt += 1
        old_loss = new_loss
        wX = np.zeros((size, 1))
        wX = np.dot(w, X) #初始化wX，方便后续计算sigmoid函数
        gradient_w = - np.dot(X, (Y - sigmoid(wX))) /size #归一化处理 3x1
        old_w = w
        w = w - alpha * lamuda * w - alpha * gradient_w.T #迭代w的值
        new_loss = loss(X, Y, w, lamuda)
        if old_loss < new_loss: #步长过大，下降不了
            w = old_w
            alpha /= 2
            continue
        if old_loss - new_loss < epsilon: #结束迭代
        #if np.linalg.norm(gradient_w) <= epsilon:
            break
    print(cnt)
    return w

# 共轭梯度法
def conjugate_gradient(X, Y, w, epsilon, lamuda=0):
    cnt = 0
    Q = np.dot(X, X.T) + lamuda * np.eye(X.shape[0])
    w = np.zeros((1, X.shape[0]))
    gradient_w = derivative(w, X, Y, lamuda)
    # gradient_w = np.dot(w, X).dot(X.T) - np.dot(X, Y).T + lamuda * w # 3x2n,2nx3,1x3
    r = -gradient_w
    d = r
    # for i in range(X.shape[0]):
    # while np.linalg.norm(gradient_w) >= 0.1 :
    while cnt < X.shape[0]:
        cnt += 1
        alpha = np.dot(r, r.T) / np.dot(d, Q).dot(d.T)
        r_old = r
        w = w + alpha * d
        r = r - alpha * np.dot(d, Q)
        beta = np.dot(r, r.T) / np.dot(r_old, r.T)
        d = r + beta * d
    print(cnt)
    # cnt = 0 #限制迭代次数
    # size = X.shape[1]
    # wX = np.zeros((size, 1))
    # wX = np.dot(w, X)
    # A = np.dot(X, X.T) #方便计算，为原式的正定矩阵 3x3
    # gradient_w = derivative(w, X, Y, lamuda) #计算第一次梯度方向 3x1
    # d = - gradient_w #第一次迭代方向为负梯度方向 3x1
    # alpha = - np.dot(d.T, gradient_w) / np.dot(d.T, A).dot(d) #初始化步长
    # while cnt<=2*size: #控制迭代次数为w的维数
    #     alpha = - np.dot(d.T, gradient_w) / np.dot(d.T, A).dot(d) #更新步长，使损失函数达到最小的步长
    #     w = w + alpha * d.T #更新w矩阵，沿着共轭方向下降
    #     gradient_w = np.dot(X, (Y - sigmoid(wX))) #更新梯度，计算共轭方向和步长需要
    #     beta = np.dot(d.T, A).dot(gradient_w) / np.dot(d.T, A).dot(d) #计算共轭方向需要的线性关系系数
    #     d = -gradient_w + beta * d #得到共轭方向
    #     cnt = cnt + 1
    # print(cnt)
    return w


       # 1x2n,2nx3,3x2n,2nx1  3x2n,2nx3,1x2n,2nx1   1x3 3x3
       # 3x2n, 2nx1

# 牛顿法
def newton(X, Y, w, epsilon, lamuda=0):
    cnt = 0
    size = X.shape[1]
    I = np.eye(X.shape[0])
    wX = np.zeros((size, 1))
    wX = np.dot(w, X)
    while cnt < 1000:
        cnt += 1
        gradient_w = np.dot(X, (Y - sigmoid(wX))) #1x3
        # print(np.linalg.norm(gradient_w))
        gradient2_w = np.dot(X, X.T) * np.dot(sigmoid(wX).T,sigmoid(-wX)) + lamuda * I
        w = w - np.dot(gradient_w.T, np.linalg.inv(gradient2_w))
        if np.linalg.norm(gradient_w) <= epsilon:
            break
    print(cnt)
    return w

#一阶导
def derivative(w, X, Y, lamuda=0):
    result = np.zeros((1, X.shape[0]))
    for i in range(X.shape[1]):
        multi = np.dot(w, X[:, i])
        result += (Y[i] - math.exp(multi) / (1 + math.exp(multi))) * X[:, i].T
    return -1 * result + lamuda * w

#二阶导（海瑟阵）
def second_derivative(w, X, lamuda=0):
    result = np.eye(X.shape[0]) * lamuda
    for i in range(X.shape[1]):
        matrix = X[:, i].T.reshape(1, X.shape[0])
        multi = np.dot(w, X[:, i])
        r = math.exp(multi) / (1+math.exp(multi))
        result += np.dot(matrix.T, matrix) * r * (1-r)
    return np.linalg.pinv(result)

#牛顿法
def newton2(X, Y, w, epsilon, lamuda=0):
    cnt = 0
    while True:
        cnt += 1
        gradient = derivative(w, X, Y, lamuda)
        if np.linalg.norm(gradient) < epsilon:
            break
        w -= np.dot(gradient, second_derivative(w, X, lamuda))
    print(cnt)
    return w


lamuda = 1
epsilon = 1e-3
alpha = 0.1
train_sample = 1000
accept_gradient = 0.1 #梯度下降法阈值
loss_1 = 0
X, Y = generate_data(train_sample, True)
train_X = np.ones((X.shape[0] + 1, 2 * train_sample)) #初始化训练样本
train_X[1:X.shape[0]+1, :] = X #训练样本变成增广矩阵
w = np.zeros((1, X.shape[0] + 1)) #初始化w
# w1 = np.zeros((1, X.shape[0] + 1)) #初始化w1(带正则项)

# 得到w
w = gradient_descent(train_X, Y, w, alpha, epsilon)
# w1 = gradient_descent(train_X, Y, w, alpha, epsilon, lamuda)

# w = newton(train_X, Y, w, epsilon, lamuda)
# w1 = newton(train_X, Y, w, epsilon, lamuda)

# w = conjugate_gradient(train_X, Y, w, epsilon, lamuda)
# w1 = conjugate_gradient(train_X, Y, w, epsilon, lamuda)

w = newton2(train_X, Y, w, epsilon)
# w1 = newton2(train_X, Y, w, epsilon, lamuda)

w = w[0]
# w1 = w1[0]


# print(loss)
# print(train_X)

# 得到回归面
print(w)
# print(w1)
test_x = np.linspace(-2, 2)
test_y = - (w[0] + w[1] * test_x) / w[2]
# test_y_1 = - (w1[0] + w1[1] * test_x) / w1[2]
plt.plot(test_x, test_y, 'r')
# plt.plot(test_x, test_y_1, 'g')
plt.show()

