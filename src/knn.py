import random
import operator
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 100个正态分布的悲伤
grief_heights = np.random.normal(50, 6, 100)
grief_weights = np.random.normal(5, 0.5, 100)

# 100个正态分布的痛苦
agony_heights = np.random.normal(30, 6, 100)
agony_weights = np.random.normal(4, 0.5, 100)

# 100个正态分布的绝望
despair_heights = np.random.normal(45, 6, 100)
despair_weights = np.random.normal(2.5, 0.5, 100)

plt.rcParams['font.sans-serif'] = ['SimHei']
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10

plt.scatter(40,2.7, c='r', s=200, marker='*', alpha=0.8)
plt.scatter(grief_heights, grief_weights, c='g', marker='s', s=50, alpha=0.8)
plt.scatter(agony_heights, agony_weights, c='b', marker='^', s=50, alpha=0.8)
plt.scatter(despair_heights, despair_weights, c='y', s=50, alpha=0.8)
plt.axis((10, 70, 1, 7))
plt.xlabel('身高(cm)', size=15)
plt.ylabel('体重(kg)', size=15)
plt.show()










