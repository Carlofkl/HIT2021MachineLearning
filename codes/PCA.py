'''
(1)首先人工生成一些数据（如三维数据），让它们主要分布在低维空间中，如首先让某个维度的方差
远小于其它维度，然后对这些数据旋转。生成这些数据后，用你的PCA方法进行主成分提取。
(2)找一个人脸数据（小点样本量），用你实现PCA方法对该数据降维，找出一些主成分，然后用这
些主成分对每一副人脸图像进行重建，比较一些它们与原图像有多大差别（用信噪比衡量）。
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D

'''
PCA 降维
data 数量*维度 nxd
data_mean 样本均值
eigValInd 排序好的特征值 dxk
redEigVects 降维后的特征值
'''
def PCA(data, k = 2):
    rows, cols = data.shape
    data_mean = np.sum(data, 0) / rows
    data_center = data - data_mean # 中心化 nxd
    covMat = np.dot(data_center.T, data_center) # dxd
    eigVals, eigVects = np.linalg.eig(covMat) # 求协方差矩阵特征值和特征向量，特征向量按列读取dxd
    eigValInd = np.argsort(eigVals) # 特征值排序
    redEigVects = eigVects[: ,eigValInd[: -(k+1): -1]] # 序列逆向排列，取前k个特征向量dxk，
    redEigVects = np.real(redEigVects) # 如果出现复向量，对其保留实部
    data_tmp = np.dot(data_center, redEigVects) # 降维后的数据 nxk
    data_recon = np.dot(data_tmp, redEigVects.T) + data_mean # 重构后的数据 nxd
    return redEigVects, data_recon

'''
生成Swiss Roll数据 nxd
n_sample 数据点数量
noise 数据店的噪声
y_scale y方向厚度
'''
def make_swiss_roll(n_sample = 1000, noise = 0.0, y_scale = 10, degree = 45):
    t = 2 * np.pi * (1 + 2 * np.random.rand(1, n_sample)) # 定义变量
    x = t * np.cos(t) # 定义x方向数据
    y = y_scale * np.random.rand(1, n_sample)
    z = t * np.sin(t)
    data = np.concatenate((x, y, z)) # 将三维数据压缩成一个向量，即数据有三个属性
    data += noise * np.random.rand(3, n_sample) # 加入噪声
    data = rotate(data, np.pi * degree / 180, 'x') # 绕x轴旋转degree，默认45
    # show_3D(data.T)
    return data.T # 得到nxd

def rotate(data, theta = 0, axis = 'x'):
    if axis == 'x':
        rotate = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        return np.dot(rotate, data)
    elif axis == 'y':
        rotate = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        return np.dot(rotate, data)
    elif axis == 'z':
        rotate = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        return np.dot(rotate, data)
    else:
        print('ERROR asix')
        return X

'''
生成二维高斯数据
n_sample 样本数量
data 维度 * 数目的矩阵
'''
def make_2D_gaussian(n_sample = 100):
    mean = [-3, 4]
    cov = [[1, 0], [0, 0.01]]
    data = np.random.multivariate_normal(mean, cov, n_sample).T
    return data

'''
生成数据，人脸图像
图片尺寸 300x300
7张图片
返回 7x90000
'''
def read_faces(file_path, size):
    file_list = os.listdir(file_path)
    data = []
    i = 1
    for file in file_list:
        path = os.path.join(file_path, file)
        with open(path) as f:
            img = cv2.imread(path, 0) # 参数0，默认灰度图打开
            img = cv2.resize(img, size) # 重新将图片压缩成size大小
            img_col = img.reshape(img.shape[0] * img.shape[1])
            data.append(img_col) # 得到数据集
    return np.array(data)

'''
计算信噪比
'''
def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

'''
展示图像 
'''
def show_3D(x):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev = 20, azim = 80)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 0], cmap=plt.cm.gnuplot)
    ax.legend(loc = 'best')
    plt.show()
def show_2D(x):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=80)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 0], cmap=plt.cm.gnuplot)
    ax.legend(loc = 'best')
    plt.show()
def show_1D(x):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=80)
    ax.scatter(x[:,0], x[:,1], x[:,2], c = x[:,0], cmap = plt.cm.gnuplot)
    ax.legend(loc = 'best')
    plt.show()

def show_3D_to_1D(data_1, data_2, data_3):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 6), facecolor='w')
    # cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
    # cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.01, top=0.99)


    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], c=data_1[:, 0], cmap=plt.cm.gnuplot)
    ax1.view_init(elev=15, azim=55)
    plt.title('原始三维数据图')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.grid(True)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], c=data_2[:, 0], cmap=plt.cm.gnuplot)
    ax2.view_init(elev=15, azim=55)
    plt.title('降至二维图')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.grid(True)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], c=data_3[:, 0], cmap=plt.cm.gnuplot)
    ax3.view_init(elev=15, azim=55)
    plt.title('降至一维图')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.grid(True)

    plt.show()

'''
主函数1：
人工生成数据 nxd
PCA降维，从3维到2维到1维
输出2维和1维时候特征向量
'''
data_1 = make_swiss_roll()
redEigVects_2, data_2 = PCA(data_1)
redEigVects_3, data_3 = PCA(data_2, 1)
print('**************************************************************************************')
print('二维特征向量')
print(redEigVects_2)
print('一维特征向量')
print(redEigVects_3)
show_3D_to_1D(data_1, data_2, data_3)
print('**************************************************************************************')

'''
主函数2：
人脸数据 nx1600
PCA降维，从1600维到5维到3维到1维
输出2维和1维时候特征向量
'''
size = (40, 40)
data = read_faces('temp', size)
data = read_faces('PCA_FACES', size)
n_sample = data.shape[0]
n_dimension = data.shape[1]
redEigVects_1, data_recon_1 = PCA(data, 5)
redEigVects_2, data_recon_2 = PCA(data, 3)
redEigVects_3, data_recon_3 = PCA(data, 1)
print('--------特征向量如下--------')
print("降至5维时：")
print(redEigVects_1)
print("降至3维时：")
print(redEigVects_2)
print("降至1维时：")
print(redEigVects_3)
print("---------信噪比如下---------")
print("降至5维时：")
for i in range(n_sample):
    print('图', i, '的信噪比为：', psnr(data[i], data_recon_1[i]))
print("降至3维时：")
for i in range(n_sample):
    print('图', i, '的信噪比为：', psnr(data[i], data_recon_2[i]))
print("降至1维时：")
for i in range(n_sample):
    print('图', i, '的信噪比为：', psnr(data[i], data_recon_3[i]))



'''
展示人脸图像
3次降维图
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize = (n_sample*1.5, 4*2))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
for i in range(n_sample):
    fig.add_subplot(4, n_sample, i+1)
    plt.title('原图')
    plt.imshow(data[i].reshape(size), cmap='gray')
    fig.add_subplot(4, n_sample, i+1+n_sample)
    plt.title('五维重构')
    plt.imshow(data_recon_1[i].reshape(size), cmap='gray')
    fig.add_subplot(4, n_sample, i+1+n_sample*2)
    plt.title('三维重构')
    plt.imshow(data_recon_2[i].reshape(size), cmap='gray')
    fig.add_subplot(4, n_sample, i+1+n_sample*3)
    plt.title('一维重构')
    plt.imshow(data_recon_3[i].reshape(size), cmap='gray')
plt.show()