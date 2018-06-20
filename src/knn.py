import random
import operator
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.KdTree import KdTree

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

# 设置样本集
grieves = map(lambda x,y:tuple(((x,y),'g')), grief_heights, grief_weights)
agonies = map(lambda x,y:tuple(((x,y),'b')), grief_heights, grief_weights)
despairs = map(lambda x,y:tuple(((x,y),'y')), grief_heights, grief_weights)
# 创建kd树
tree = KdTree(list(grieves)+list(agonies)+list(despairs))
# 穷举生成空间上的点
all_points = []
for i in range(100, 701, 10):
    for j in range(100, 701, 10):
        all_points.append((float(i)/10., float(j)/100.))
# 一共36万个点
len(all_points)
# 设置归一化距离函数
def normalized_dist(x,y):
    return (x[0]-y[0])**2+(10*x[1]-10*y[1])**2

# 每个点运算15NN，并记录计算时间
now = datetime.datetime.now()
fifteen_NN_result = []
for point in all_points:
    fifteen_NN_result.append((point, tree.kNN(point, k=15, dist=normalized_dist)[0]))
print(datetime.datetime.now()-now)

# 把每个颜色的数据分开
fifteen_NN_yellow = []
fifteen_NN_green = []
fifteen_NN_blue = []

for pair in fifteen_NN_result:
    if pair[1] == 'y':
        fifteen_NN_yellow.append(pair[0])
    if pair[1] == 'g':
        fifteen_NN_green.append(pair[0])
    if pair[1] == 'b':
        fifteen_NN_blue.append(pair[0])

plt.scatter(40, 2.7, c='r', s=200, marker='*', alpha=0.8, zorder=10)
plt.scatter(grief_heights, grief_weights, c='g', marker='s', s=50, alpha=0.8, zorder=10)
plt.scatter(agony_heights, agony_weights, c='b', marker='^', s=50, alpha=0.8, zorder=10)
plt.scatter(despair_heights, despair_weights, c='y', s=50, alpha=0.8, zorder=10)
# plt.scatter([x[0] for x in fifteen_NN_yellow],[x[1] for x in fifteen_NN_yellow], s=50, c='yellow', marker='1', alpha=0.8)
# plt.scatter([x[0] for x in fifteen_NN_blue],[x[1] for x in fifteen_NN_blue], s=50, c='blue', marker='2', alpha=0.8)
# plt.scatter([x[0] for x in fifteen_NN_green],[x[1] for x in fifteen_NN_green], s=50, c='green', marker='3', alpha=0.8)
plt.axis((10, 70, 1, 7))
plt.title('15NN分类', size=30)
plt.xlabel('身高(cm)', size=15)
plt.ylabel('体重(kg)', size=15)

plt.show()






















