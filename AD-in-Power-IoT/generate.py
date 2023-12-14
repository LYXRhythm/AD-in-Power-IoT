import copy
import random

import numpy as np

from utils import *

generation = 1800
energy = np.zeros((generation, N + 1))

for i in range(generation):
    fs = 1000
    n = 1000
    t = np.arange(0, n / fs, 1 / fs)
    freq = 10
    noise = 1.2*np.random.randn(np.size(t))
    sus = copy.deepcopy(sus_const_5_5)

    for j in range(N):
        random_x = round(random.uniform(-1, 1), 3)
        random_y = round(random.uniform(-1, 1), 3)
        sus[j][0] = sus_const_1010[j][0] + 400 * random_x
        sus[j][1] = sus_const_1010[j][1] + 400 * random_y

        dist1[j] = math.pow((math.pow(sus[j][0] - pus2[0][0], 2) + math.pow(sus[j][1] - pus2[0][1], 2)), 0.5)
        dist2[j] = math.pow((math.pow(sus[j][0] - pus2[1][0], 2) + math.pow(sus[j][1] - pus2[1][1], 2)), 0.5)

    s1 = random.choice([0, 1])
    s2 = random.choice([0, 1])

    signal1 = add_noise(0.6 * np.sin(2 * np.pi * freq * t), snr_db, s1)
    signal2 = add_noise(0.6 * np.cos(2 * np.pi * freq * t), snr_db, s2)
    x1 = signal1 + noise
    x2 = signal2 + noise
    label = s1 or s2
    energy[i][N] = label

    # ==IGNORE ME== threshold = np.mean(np.dot(np.abs(x1), np.abs(x1)) + np.dot(np.abs(x2), np.abs(x2)))
    for j in range(N):
        energy[i][j] = np.dot(np.abs(x1), np.abs(x1)) / dist1[j] + np.dot(np.abs(x2), np.abs(x2)) / dist2[j]

energy = np.round(energy, 8)
np.savetxt('./dataset/dataset_5_5_dynamic.txt', energy, fmt='%.8f', delimiter=',')  # FIXME: 每次生成数据时需要修改文件名

