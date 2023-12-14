from globals import *
import numpy as np
import pandas as pd


# 向原始信号添加定量信噪比噪声
def add_noise(signal, snr_db, state):
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    # 将信噪比转化为线性信噪比
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    # 添加噪声后的信号
    noisy_signal = state * signal + noise
    return noisy_signal


# 从txt文件中读取数据
def read_data_from_txt(file):
    data = []
    with open(file, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data.append(row)
    return data

def split_data(data):  # 将数据分割为data和label
    data = pd.DataFrame(data)
    label = data[:][N]
    data = data[data.columns[0:N]]
    data = np.array(data)
    label = np.array(label)
    return data, label

def split_data2(data, n):  # 将数据分割为data和label
    data = pd.DataFrame(data)
    label = data[:][n]
    data = data[data.columns[0:n]]
    data = np.array(data)
    label = np.array(label)
    return data, label

def train(clf, x_train, y_train):
    clf.fit(x_train, y_train.ravel())   # 训练集特征向量和 训练集标签值

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy: %.4f' % (tip, np.mean(acc)))

def print_accuracy(clf, x_train, y_train, x_test, y_test, tip):
    # 分别打印训练集和测试集的准确率 score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率
    print('%s training prediction: %.4f' % (tip, clf.score(x_train, y_train)))
    print('%s test dataset prediction: %.4f' % (tip, clf.score(x_test, y_test)))
    # 原始结果和预测结果进行对比 predict() 表示对x_train样本进行预测,返回样本类别
    show_accuracy(clf.predict(x_train), y_train, tip + ' training dataset')
    show_accuracy(clf.predict(x_test), y_test, tip + ' testing dataset')
