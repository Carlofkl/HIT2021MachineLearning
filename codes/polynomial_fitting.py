'''
多项式拟合
'''
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def generate_data(m, size=10, var=0.25, mean = 0, begin=0, end=1):
    x = np.linspace(begin, end, size) #在[begin, end]生成size大小的数组
    Y = np.sin(2 * np.pi * x) #生成样本y，未加入高斯噪声
    for i in range(size):
        Y[i] += np.random.normal(mean, var) #加入高斯噪声
    Y = Y.T # Y需要转置
    X = np.zeros((m+1, size)) #初始化X矩阵
    plt.plot(x, Y, 'bo') #绘制测试样本
    for i in range(m+1):
        X[i] = x ** i #得到每一行的X
    X = X.T #X需要转置
    # print(x)
    # print(X)
    # print(y)

    return X, Y

#损失函数，不带正则项
def loss_1(X, Y):
    w = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    return w

#损失函数，有正则项
def loss_2(X, Y, lamuda):
    w = np.linalg.inv(np.dot(X.T, X) + lamuda).dot(X.T).dot(Y)
    return w

#梯度下降法
def gradient_descent(X, Y, m, alpha, accept_gradient, lamuda):
    cnt = 0 #记录迭代次数
    w = np.zeros(m+1).T #初始化w矩阵
    gradient_w = np.dot(X.T, X).dot(w) - np.dot(X.T, Y) + lamuda * w #计算初始梯度，迭代方向为负梯度方向
    while np.linalg.norm(gradient_w) >= accept_gradient:
        cnt = cnt + 1
        w = w - alpha * gradient_w #? 更新w矩阵
        gradient_w = np.dot(X.T, X).dot(w) - np.dot(X.T, Y) + lamuda * w #更新梯度方向
    print(cnt)
    return w

#共轭梯度法
def conjugate_gradient(X, Y, m, accept_gradient):
    cnt = 0 #限制迭代次数
    w = np.zeros(m+1).T #初始化需要求解的w
    A = np.dot(X.T, X) #方便计算，为原式二次型的正定矩阵
    gradient_w = np.dot(A, w) - np.dot(X.T, Y) + lamuda * w #计算第一次梯度方向
    d = - gradient_w #第一次迭代方向为负梯度方向
    alpha = np.dot(A, w) - np.dot(X.T, Y) #初始化步长
    while cnt <= m: #控制迭代次数为w的维数
        alpha = - np.dot(gradient_w.T, d) / np.dot(d.T, A).dot(d) #更新步长，使损失函数达到最小的步长
        w = w + alpha * d #更新w矩阵，沿着共轭方向下降
        gradient_w = np.dot(A, w) - np.dot(X.T, Y) + lamuda * w #更新梯度，计算共轭方向和步长需要
        beta = np.dot(d.T, A).dot(gradient_w) / np.dot(d.T, A).dot(d) #计算共轭方向需要的线性关系系数
        d = -gradient_w + np.dot(beta, d) #得到共轭方向
        cnt = cnt + 1
    print(cnt)
    return w




'''
主函数
'''
m = 5 #阶数
lamuda = 0 #惩罚项的lamda
var = 0.25 #高斯噪声的方差
alpha = 0.01 #梯度下降法的步长
accept_gradient = 0.1 #梯度下降法阈值
train_sample = 10 #训练样本个数
test_sample = 100 #测试样本个数
test_x = np.linspace(0, 1, test_sample)
test_X = np.zeros((m+1, test_sample))
for i in range(m+1):
    test_X[i] = test_x ** i #生成测试样本X

x_t = np.arange(0, 1, 0.01)
y_t = np.sin(2 * np.pi * x_t)
plt.title('green-sin(2pix), blue-train_sample, red-test_sample') #绘制sin（2pix）
plt.plot(x_t, y_t, 'g')

X, Y = generate_data(m, train_sample, var) #生成数据

#不同方法得到w
w = loss_1(X, Y) #最小二乘法，不带正则项
# w = loss_2(X, Y, lamuda) #最小二乘法，带正则项
# w = gradient_descent(X, Y, m, alpha, accept_gradient, lamuda) #梯度下降法
# w = conjugate_gradient(X, Y, m, accept_gradient)

#展示
print(w)
test_y = np.dot(w.T, test_X)
plt.plot(test_x, test_y, 'r')
plt.show()







