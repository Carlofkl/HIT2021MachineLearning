import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_uci():
    path = "Exasens.csv"
    data_set = pd.read_csv(path)
    x = data_set['Diagnosis']
    y = data_set.drop('Diagnosis', axis=1)
    _label, _data = np.array(x, dtype=str), np.array(y, dtype=int)
    for i in range(_label.shape[0]):
        if _label[i] == "COPD":
            _label[i] = 0
        elif _label[i] == "HC":
            _label[i] = 1
        elif _label[i] == "Asthma":
            _label[i] = 2
        elif _label[i] == "Infected":
            _label[i] = 3
    labels = np.array(_label, dtype=int)
    print()
    return labels, _data


def generate_data(k, count):
    means = np.zeros((k, 2))                   # 各个高斯分布的均值
    _data = np.zeros((count, 3))                 # 待分类的数据
    for i in range(k):
        means[i, 0] = float(input("输入第 " + str(i + 1) + " 组数据的均值\n"))
        means[i, 1] = float(input())
    cov = np.array([[1, 0], [0, 1]])            # 协方差矩阵
    index = 0
    for i in range(k):
        for j in range(int(count / k)):
            _data[index, 0:2] = np.random.multivariate_normal(means[i], cov)
            _data[index, 2] = i
            index += 1
    return means, _data


# 找到最小距离
def get_min(distance):
    value = distance[0, 0]
    index = 0
    for i in range(1, distance.shape[1]):
        if value > distance[0, i]:
            value = distance[0, i]
            index = i
    return index


# k-means算法
def k_means(sample, num):
    cnt= -1                           # 记录迭代次数
    dimension = sample.shape[1]             # 数据的维度
    all_list = []                                   # 存放k个簇
    center = np.zeros((num, dimension))         # 记录k个簇的中心
    maxi = len(sample)
    rand = np.random.randint(0, maxi, num)      # 随机选取k个样本初始化
    for i in range(num):
        temp_list = [rand[i]]
        all_list.append(temp_list)
    while True:                           # 迭代
        cnt += 1
        old_center = center.copy()          # 记录更新之前的中心
        for i in range(num):
            cluster = all_list[i]
            length = len(cluster)
            sums = np.zeros((1, dimension))    # 记录一个簇所有数据各维度的加和
            for j in range(dimension):       # 计算每个簇的中心
                for f in range(length):
                    sums[0, j] += sample[cluster[f], j]
                center[i, j] = float(sums[0, j] / length)  # 得到中心的各个维度
        if np.sum(abs(center - old_center)) < 0.1:       # center基本不变，结束迭代
            print(cnt)
            return center, all_list
        for i in range(num):
            all_list[i].clear()
        index = -1
        for info in sample:
            index += 1
            distance = np.zeros((1, num))  # 记录一个数据到三个中心的距离
            for i in range(num):
                for j in range(dimension):
                    distance[0, i] += float((center[i, j] - info[j]) ** 2)
            category = get_min(distance)        # 找到距离该数据最近的中心
            all_list[category].append(index)     # 划分簇


# 初始化先验概率、均值和协方差矩阵
def init(center, sample, num):
    dimension = sample.shape[1]
    # u = np.zeros((num, dimension))       # 初始化均值
    u = center.copy()
    covariance = np.eye(dimension)    # 初始化协方差
    cov = []
    alpha = np.zeros((num, 1))          # 初始化先验
    # maxi = len(sample)
    # rand = np.random.randint(0, maxi, num)  # 随机选取k个样本初始化
    for i in range(num):
        # u[i, :] = sample[rand[i]]
        alpha[i, 0] = float(1 / num)
        cov.append(covariance)
    return u, alpha, cov


def gaussian_probability(x, u, covariane, num):
    delta = x.reshape(len(x), 1) - u.reshape(len(u), 1)
    covariane1 = pow(np.linalg.det(covariane), 0.5)
    index1 = -num / 2
    pai = pow((2 * np.pi), index1)
    index2 = (-1/2) * np.dot(delta.T, np.linalg.inv(covariane)).dot(delta)
    prior = (covariane1 * pai * np.exp(index2))
    return prior


# 通过最大似然得到均值、协方差矩阵和先验概率
def like_hood(sample, gama, num):
    dimension = sample.shape[1]
    number = sample.shape[0]
    u = np.zeros((num, dimension))      # 更新均值
    cov = []                            # 更新协方差矩阵
    alpha = np.zeros((num, 1))          # 更新先验
    # 更新均值
    for i in range(num):
        for j in range(dimension):
            divisor = 0
            dividend = 0
            for f in range(number):
                divisor += gama[f, i]
                dividend += gama[f, i] * sample[f, j]
            u[i, j] = float(dividend / divisor)
    # 更新协方差矩阵
    for i in range(num):
        divisor = 0
        dividend = np.zeros((dimension, dimension))
        for j in range(number):
            delta = sample[j].reshape(dimension, 1) - u[i, :].reshape(dimension, 1)
            matrix = np.dot(delta, delta.T)
            dividend += gama[j, i] * matrix
            divisor += gama[j, i]
        covariance = dividend / divisor
        cov.append(covariance)
    # 更新先验概率
    for i in range(num):
        gama_sum = 0
        for j in range(number):
            gama_sum += gama[j, i]
        alpha[i, 0] = float(gama_sum / number)
    return u, cov, alpha


# 通过均值、协方差矩阵和先验概率求得后验概率
def gaussian_mixture(sample, num, u, alpha, cov):
    number = sample.shape[0]
    gama = np.zeros((number, num))
    total_probability = []                            # 全概率
    for j in range(number):
        sums = 0
        for i in range(num):
            sums += float(alpha[i, 0] * gaussian_probability(sample[j, :], u[i, :], cov[i], num))
        total_probability.append(sums)              # 计算全概率

    for j in range(number):
        for i in range(num):                        # 计算xj属于i类的后验概率
            gama[j, i] = float(alpha[i, 0] * gaussian_probability(sample[j, :], u[i, :], cov[i], num)
                               / total_probability[j])
    return gama


# EM算法
def em_algorithm(center, sample, k):
    u, alpha, cov = init(center, sample, k)   # 初始化参数
    iterator = 0                        # 记录迭代次数
    while True:
        iterator += 1                   # 迭代次数
        prev_u = u.copy()               # 记录上一次迭代的参数
        # prev_alpha = alpha.copy()
        # prev_cov = cov.copy()
        gama = gaussian_mixture(sample, k, u, alpha, cov)     # E步

        u, cov, alpha = like_hood(sample, gama, k)                 # M步
        if np.sum(abs(prev_u - u)) < 0.05:  # 均值基本不变，结束迭代
            print(iterator - 1)
            break
    cluster = classify(sample, k, u, cov, alpha)
    return u, cov, alpha, cluster


# 划分簇
def classify(sample, num, u, cov, alpha):
    all_list = []
    for i in range(num):
        temp = []
        all_list.append(temp)                   # 创建num个簇
    gama = gaussian_mixture(sample, num, u, alpha, cov)
    for i in range(gama.shape[0]):
        index = 0
        maxi = gama[i, 0]
        for j in range(1, gama.shape[1]):
            if gama[i, j] > maxi:
                maxi = gama[i, j]
                index = j                       # 寻找最大的gama来划分簇
        all_list[index].append(i)
    return all_list


# 获得最大数的索引
def get_attribute(label_set):
    maxi = label_set[0, 0]
    for i in range(1, label_set.shape[0]):
        if maxi < label_set[i, 0]:
            maxi = label_set[i, 0]
    return maxi


# 准确率分析
def accuracy(cluster, labels, num):
    sum_classified = 0
    for i in range(k):
        label_set = np.zeros((num, 1))
        for index in cluster[i]:
            index1 = int(labels[index])
            label_set[index1, 0] += 1
        sum_classified += get_attribute(label_set)
    my_accuracy = float(sum_classified / labels.shape[0])
    return my_accuracy


# 绘图
def draw_point(point_set, num, center, accurate):
    count = int(len(point_set) / num)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("accuracy = " + str(accurate))
    for i in range(num):
        up = (i + 1) * count
        down = i * count
        set_x = point_set[down: up, 0]
        set_y = point_set[down: up, 1]
        plt.plot(set_x, set_y, linestyle='', marker='.')
    for i in range(num):
        plt.plot(center[i, 0], center[i, 1], linestyle='', marker='+', color='Black')
    plt.show()


if __name__ == '__main__':
    k = 4                   # 高斯分布个数
    data_num = 1000         # 待分类的数据集大小


    # label, data = get_uci()
    # center1, cluster1 = k_means(data, data.shape[1])
    # center2, covariances, alphas, cluster2 = em_algorithm(center1, data, data.shape[0])


    mean1, data_label = generate_data(k, data_num)

    label1 = data_label[:, 2].reshape(1, data_num)
    label = np.array(label1[0], dtype=int)

    data = data_label[:, 0:2].reshape(data_num, 2)

    centers1, clusters1 = k_means(data, k)
    print(mean1)
    print(centers1)

    centers2, covariances, alphas, clusters2 = em_algorithm(centers1, data, k)
    print('中心：')
    print(centers2)
    print('alpha：')
    print(alphas)
    print('协方差矩阵：')
    for covs in covariances:
        print(covs)

    draw_point(data, k, centers1, accuracy(clusters1, label, k))
    draw_point(data, k, centers2, accuracy(clusters2, label, k))
