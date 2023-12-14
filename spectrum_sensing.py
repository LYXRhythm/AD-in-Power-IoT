from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from globals import *
from utils import *

energy = np.array(read_data_from_txt('dataset_5_5_dynamic.txt'), dtype=float)
# 划分数据集
train_data = energy[:1200]
train_data, train_label = split_data(train_data)

test_data = energy[1200:]
test_data, test_label = split_data(test_data)


"""单独训练分类器"""
# SVM
svm = SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
train(svm, train_data, train_label)
print_accuracy(svm, train_data, train_label, test_data, test_label, 'SVM')
print('---------------------------------------------')
# 贝叶斯
gnb = GaussianNB()
train(gnb, train_data, train_label)
print_accuracy(gnb, train_data, train_label, test_data, test_label, 'Naive Bayes')
print('---------------------------------------------')
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
train(knn, train_data, train_label)
print_accuracy(knn, train_data, train_label, test_data, test_label, 'KNN')
print('---------------------------------------------')

"""粒子群优化算法求三种分类器的权重"""
"""多种分类器集成学习"""
print('---------------------------------------------')
vot = VotingClassifier(estimators=[
    ('svm_clf', SVC(C=0.8, kernel='linear', decision_function_shape='ovr', probability=True))
    , ('gnb_clf', GaussianNB())
    , ('knn_clf', KNeighborsClassifier(n_neighbors=5))
], voting='soft')
train(vot, train_data, train_label)
print_accuracy(vot, train_data, train_label, test_data, test_label, 'soft vote')
