import copy
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from Outlier.FPRW import FPRW
from utils import *

warnings.filterwarnings("ignore")

dynamic_datasets = ['5_5', '6_6', '7_diamond', '4_6', '5_7', '4_7_triangle']
dataset_pre_str = '../dataset/dataset_'
dataset_post_str = '_dynamic.txt'

false_size = 90
alter_size = 5
scalar = MinMaxScaler()

sigmas = np.arange(0, 1, 0.05)
auc_scores = np.zeros((len(dynamic_datasets), len(sigmas)))  # (6, 20)

for index1, dynamic_dataset in enumerate(dynamic_datasets):
    print(index1)
    dataset = dataset_pre_str + dynamic_dataset + dataset_post_str
    energy = np.array(read_data_from_txt(dataset), dtype=float)

    train_data_, _ = split_data2(energy, energy.shape[1]-1)

    for index2, sigma in enumerate(sigmas):
        train_data = copy.deepcopy(train_data_)

        random_index = np.sort(np.random.choice(range(len(train_data)), size=false_size, replace=False))

        # Falsification
        for i in range(false_size):
            alter_index = np.sort(np.random.choice(range(energy.shape[1]-1), size=alter_size, replace=False))  # 篡改点的索引
            for j in range(alter_size):
                train_data[random_index[i]][alter_index[j]] = train_data[random_index[i]][alter_index[j]] * 1.35

        train_data = scalar.fit_transform(train_data)
        anormal_scores_FPRW = FPRW(train_data, sigma)

        y = np.zeros(len(anormal_scores_FPRW)).astype(int)
        y[random_index] = 1

        fpr, tpr, _ = roc_curve(y, anormal_scores_FPRW)
        auc = roc_auc_score(y, anormal_scores_FPRW)
        auc_scores[index1][index2] = auc

# 画图
plt.rcParams['font.family'] = ['Times New Roman']
fig = plt.figure()
fig.set_size_inches(8, 6)
fig.set_dpi(150)

plt.plot(sigmas, auc_scores[0], 'crimson', marker='o', label='Elec55d5')
plt.plot(sigmas, auc_scores[1], '#9400D3', marker='v', label='Elec66d5')
plt.plot(sigmas, auc_scores[2], '#4A774A', marker='^', label='Elec7did5')
plt.plot(sigmas, auc_scores[3], 'navy', marker='s', label='Elec46d5')
plt.plot(sigmas, auc_scores[4], 'gold', marker='s', label='Elec57d5')
plt.plot(sigmas, auc_scores[5], 'lawngreen', marker='+', label='Elec47td5')

plt.legend(loc='lower right', fontsize=14)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('ε', fontdict={'size': 18}, labelpad=5)
plt.ylabel('AUC', fontdict={'size': 18}, labelpad=5)
plt.grid(linestyle='-.')
plt.grid(True)
plt.savefig('./figs/parameter_analysis.png', bbox_inches='tight', dpi=600)
plt.show()
