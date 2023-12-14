import copy
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.dif import DIF
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.rod import ROD
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from Outlier.FPRW import FPRW
from utils import *

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
energy = np.array(read_data_from_txt('../dataset/dataset_5_7_dynamic.txt'), dtype=float)
# g1数据用于FPRW算法
train_data_g1_, _ = split_data(energy)
train_data_g1 = copy.deepcopy(train_data_g1_)
# g2数据用于pyod包中的算法
train_data_g2_ = energy[:1200]
test_data_g2_ = energy[1200:]
train_data_g2_, _ = split_data(train_data_g2_)
test_data_g2_, _ = split_data(test_data_g2_)

train_data_g2 = copy.deepcopy(train_data_g2_)
test_data_g2 = copy.deepcopy(test_data_g2_)

# 异常数据比例
# 3% (54, 18)
# 5% (90, 30)
# 10% (180, 60)
# 15% (270, 90)
# 20% (360, 120)

# FIXME
false_size_g1 = 360
false_size_g2 = 120
alter_size = 5
scalar = MinMaxScaler()

sigma_FPRW = 0.6
random_index_g1 = np.sort(np.random.choice(range(len(train_data_g1)), size=false_size_g1, replace=False))
random_index_g2 = np.sort(np.random.choice(range(len(test_data_g2)), size=false_size_g2, replace=False))

auc_FPRW = np.zeros(10)
auc_LOF = np.zeros(10)
auc_IForest = np.zeros(10)
auc_ECOD = np.zeros(10)
auc_LSCP = np.zeros(10)
auc_DIF = np.zeros(10)
auc_ROD = np.zeros(10)

for index in range(10):
    # Falsification
    for i in range(false_size_g1):
        alter_index = np.sort(np.random.choice(range(N), size=alter_size, replace=False))  # 篡改点的索引
        for j in range(alter_size):
            train_data_g1[random_index_g1[i]][alter_index[j]] = train_data_g1[random_index_g1[i]][alter_index[j]] * 1.35

    for i in range(false_size_g2):
        alter_index = np.sort(np.random.choice(range(N), size=alter_size, replace=False))  # 篡改点的索引
        for j in range(alter_size):
            test_data_g2[random_index_g2[i]][alter_index[j]] = test_data_g2[random_index_g2[i]][alter_index[j]] * 1.35

    y_g1 = np.zeros(len(train_data_g1)).astype(int)
    y_g1[random_index_g1] = 1
    y_g2 = np.zeros(len(test_data_g2)).astype(int)
    y_g2[random_index_g2] = 1

    train_data_g1 = scalar.fit_transform(train_data_g1)

    anormal_scores_FPRW = FPRW(train_data_g1, sigma_FPRW)
    auc_FPRW[index] = roc_auc_score(y_g1, anormal_scores_FPRW)

    clf_LOF = LOF()
    clf_LOF.fit(train_data_g2)
    anornal_scores_LOF = clf_LOF.decision_function(test_data_g2)
    fpr_LOF, tpr_LOF, _ = roc_curve(y_g2, anornal_scores_LOF)
    auc_LOF[index] = roc_auc_score(y_g2, anornal_scores_LOF)

    clf_IForest = IForest()
    clf_IForest.fit(train_data_g2)
    anornal_scores_IForest = clf_IForest.decision_function(test_data_g2)
    fpr_IForest, tpr_IForest, _ = roc_curve(y_g2, anornal_scores_IForest)
    auc_IForest[index] = roc_auc_score(y_g2, anornal_scores_IForest)

    clf_ECOD = ECOD()
    clf_ECOD.fit(train_data_g2)
    anormal_scores_ECOD = clf_ECOD.decision_function(test_data_g2)
    fpr_ECOD, tpr_ECOD, _ = roc_curve(y_g2, anormal_scores_ECOD)
    auc_ECOD[index] = roc_auc_score(y_g2, anormal_scores_ECOD)

    clf_LSCP = LSCP([LOF(), DIF(batch_size=50, device=device)])
    clf_LSCP.fit(train_data_g2)
    anormal_scores_LSCP = clf_LSCP.decision_function(test_data_g2)
    fpr_LSCP, tpr_LSCP, _ = roc_curve(y_g2, anormal_scores_LSCP)
    auc_LSCP[index] = roc_auc_score(y_g2, anormal_scores_LSCP)

    clf_DIF = DIF(batch_size=50, device=device)
    clf_DIF.fit(train_data_g2)
    anormal_scores_DIF = clf_DIF.decision_function(test_data_g2)
    fpr_DIF, tpr_DIF, _ = roc_curve(y_g2, anormal_scores_DIF)
    auc_DIF[index] = roc_auc_score(y_g2, anormal_scores_DIF)

    clf_ROD = ROD()
    clf_ROD.fit(train_data_g2)
    anormal_scores_ROD = clf_ROD.decision_function(test_data_g2)
    fpr_ROD, tpr_ROD, _ = roc_curve(y_g2, anormal_scores_ROD)
    auc_ROD[index] = roc_auc_score(y_g2, anormal_scores_ROD)

auc_score_FPRW = np.average(auc_FPRW)
auc_score_LOF = np.average(auc_LOF)
auc_score_IForest = np.average(auc_IForest)
auc_score_ECOD = np.average(auc_ECOD)
auc_score_LSCP = np.average(auc_LSCP)
auc_score_DIF = np.average(auc_DIF)

print('auc_FPRW = %0.4f' % auc_score_FPRW)
print('auc_LOF = %0.4f' % auc_score_LOF)
print('auc_IForest = %0.4f' % auc_score_IForest)
print('auc_ECOD = %0.4f' % auc_score_ECOD)
print('auc_LSCP = %0.4f' % auc_score_LSCP)
print('auc_DIF = %0.4f' % auc_score_DIF)

for i in range(false_size_g1):
    alter_index = np.sort(np.random.choice(range(N), size=alter_size, replace=False))  # 篡改点的索引
    for j in range(alter_size):
        train_data_g1[random_index_g1[i]][alter_index[j]] = train_data_g1[random_index_g1[i]][alter_index[j]] * 1.35

for i in range(false_size_g2):
    alter_index = np.sort(np.random.choice(range(N), size=alter_size, replace=False))  # 篡改点的索引
    for j in range(alter_size):
        test_data_g2[random_index_g2[i]][alter_index[j]] = test_data_g2[random_index_g2[i]][alter_index[j]] * 1.35


y_g1 = np.zeros(len(train_data_g1)).astype(int)
y_g1[random_index_g1] = 1
y_g2 = np.zeros(len(test_data_g2)).astype(int)
y_g2[random_index_g2] = 1

train_data_g1 = scalar.fit_transform(train_data_g1)

anormal_scores_FPRW = FPRW(train_data_g1, sigma_FPRW)
roc_auc_FPRW = roc_auc_score(y_g1, anormal_scores_FPRW)
fpr_FPRW, tpr_FPRW, _ = roc_curve(y_g1, anormal_scores_FPRW)

# 以下为pyod包中的算法
clf_LOF = LOF()
clf_LOF.fit(train_data_g2)
anornal_scores_LOF = clf_LOF.decision_function(test_data_g2)
fpr_LOF, tpr_LOF, _ = roc_curve(y_g2, anornal_scores_LOF)
roc_auc_LOF = roc_auc_score(y_g2, anornal_scores_LOF)

clf_IForest = IForest()
clf_IForest.fit(train_data_g2)
anornal_scores_IForest = clf_IForest.decision_function(test_data_g2)
fpr_IForest, tpr_IForest, _ = roc_curve(y_g2, anornal_scores_IForest)
roc_auc_IForest = roc_auc_score(y_g2, anornal_scores_IForest)

clf_ECOD = ECOD()
clf_ECOD.fit(train_data_g2)
anormal_scores_ECOD = clf_ECOD.decision_function(test_data_g2)
fpr_ECOD, tpr_ECOD, _ = roc_curve(y_g2, anormal_scores_ECOD)
roc_auc_ECOD = roc_auc_score(y_g2, anormal_scores_ECOD)

clf_LSCP = LSCP([LOF(), DIF(batch_size=50, device=device)])
clf_LSCP.fit(train_data_g2)
anormal_scores_LSCP = clf_LSCP.decision_function(test_data_g2)
fpr_LSCP, tpr_LSCP, _ = roc_curve(y_g2, anormal_scores_LSCP)
roc_auc_LSCP = roc_auc_score(y_g2, anormal_scores_LSCP)

clf_DIF = DIF(batch_size=50, device=device)
clf_DIF.fit(train_data_g2)
anormal_scores_DIF = clf_DIF.decision_function(test_data_g2)
fpr_DIF, tpr_DIF, _ = roc_curve(y_g2, anormal_scores_DIF)
roc_auc_DIF = roc_auc_score(y_g2, anormal_scores_DIF)

clf_ROD = ROD()
clf_ROD.fit(train_data_g2)
anormal_scores_ROD = clf_ROD.decision_function(test_data_g2)
fpr_ROD, tpr_ROD, _ = roc_curve(y_g2, anormal_scores_ROD)
roc_auc_ROD = roc_auc_score(y_g2, anormal_scores_ROD)


# 画图
plt.rcParams['font.family'] = ['Times New Roman']
fig = plt.figure()
fig.set_size_inches(8, 6)
fig.set_dpi(300)

plt.title('ROC Curve on Elec57d20', fontdict={'size': 20}, pad=15)
plt.plot(fpr_FPRW, tpr_FPRW, 'crimson', label=u'FPRW')
plt.plot(fpr_LOF, tpr_LOF, '#9400D3', label=u'LOF')
plt.plot(fpr_IForest, tpr_IForest, '#4A774A', label=u'IForest')
plt.plot(fpr_ECOD, tpr_ECOD, 'navy', label=u'ECOD')
plt.plot(fpr_LSCP, tpr_LSCP, 'gold', label=u'LSCP')
plt.plot(fpr_DIF, tpr_DIF, 'lawngreen', label=u'DIF')
plt.plot(fpr_ROD, tpr_ROD, 'deepskyblue', label=u'ROD')

plt.legend(loc='lower right', fontsize=16)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate', fontdict={'size': 18}, labelpad=5)
plt.ylabel('True Positive Rate', fontdict={'size': 18}, labelpad=5)
plt.grid(linestyle='-.')
plt.grid(True)
plt.savefig('./figs/Elec57d20.png', bbox_inches='tight', dpi=600)
plt.show()
