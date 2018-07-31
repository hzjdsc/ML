import random
import operator
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.KdTree import KdTree

# 100个正态分布的悲伤
grief_heights = np.random.normal(50, 6, 20)
grief_weights = np.random.normal(5, 0.5, 20)

# 100个正态分布的痛苦R
agony_heights = np.random.normal(30, 6, 20)
agony_weights = np.random.normal(4, 0.5, 20)

# 100个正态分布的绝望
despair_heights = np.random.normal(45, 6, 20)
despair_weights = np.random.normal(2.5, 0.5, 20)

plt.rcParams['font.sans-serif'] = ['SimHei']
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 7

# 设置样本集
grieves = map(lambda x, y: tuple(((x, y), 'g')), grief_heights, grief_weights)
agonies = map(lambda x, y: tuple(((x, y), 'b')), agony_heights, agony_weights)
despairs = map(lambda x, y: tuple(((x, y), 'y')), despair_heights, despair_weights)

# 创建kd树
point_list = list(grieves) + list(agonies) + list(despairs)
print(len(point_list))
tree = KdTree(point_list)

# 穷举生成空间上的点
all_points = []
for i in range(100, 701, 10):
    for j in range(100, 701, 10):
        all_points.append((float(i) / 10., float(j) / 100.))

print(len(all_points))


# 设置归一化距离函数
def normalized_dist(x, y):
    return (x[0] - y[0]) ** 2 + (10 * x[1] - 10 * y[1]) ** 2


# 每个点运算15NN，并记录计算时间
start_t = datetime.datetime.now()
grief_prob = []
for point in all_points:
    grief_prob.append((point, tree.kNN_prob(point, label='g', k=5, dist=normalized_dist)))
end_t = datetime.datetime.now() - start_t

plt.scatter(40, 2.7, c='r', s=200, marker='*', alpha=0.8, zorder=10)
plt.scatter(grief_heights, grief_weights, c='g', marker='s', s=50, alpha=0.8, zorder=10)
plt.scatter(agony_heights, agony_weights, c='b', marker='^', s=50, alpha=0.8, zorder=10)
plt.scatter(despair_heights, despair_weights, c='y', s=50, alpha=0.8, zorder=10)

plt.scatter([x[0][0] for x in grief_prob], [x[0][1] for x in grief_prob],
            s=10, c=[x[1] for x in grief_prob], marker='1', cmap='Greens')

plt.axis((10, 70, 1, 7))
plt.title('5NN分类', size=30)
plt.xlabel('身高(cm)', size=15)
plt.ylabel('体重(kg)', size=15)

plt.show()
