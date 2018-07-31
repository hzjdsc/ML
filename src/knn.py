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
# grief_heights = [52.06105112, 45.29148154, 53.09354958,
#                  54.60271404, 58.8238066, 39.31671937, 50.46985025,
#                  42.96313143]
# grief_weights = [4.64533944, 4.94655363, 3.83105107,
#                  6.56714855, 5.72951018, 4.79345601, 5.26353153,
#                  4.20818748]

# 100个正态分布的痛苦R
agony_heights = np.random.normal(30, 6, 20)
agony_weights = np.random.normal(4, 0.5, 20)
# agony_heights = [24.51113812, 22.9572049, 36.18256777, 17.97887871,
#                  29.80585883, 19.5845069, 25.98263527, 30.39058502,
#                  27.82357139, 12.89093926]
# agony_weights = [5.18343006, 3.53049261, 3.61778993,
#                  2.8609095, 4.90906785, 4.79833652, 4.12361779,
#                  4.00871815, 4.34458511, 4.65541531]

# 100个正态分布的绝望
despair_heights = np.random.normal(45, 6, 20)
despair_weights = np.random.normal(2.5, 0.5, 20)
# despair_heights = [47.60772409, 43.16310204, 49.41393005, 28.21783366,
#                    49.78131412, 44.50355543, 51.92690906, 31.55062205, 45.18332532, 45.35056474,
#                    32.11078176, 36.99199215, 55.65058493, 39.79352219, 55.7779951]
# despair_weights = [2.8144508, 2.23171516, 1.9424559, 2.43355543, 3.59951873, 1.63326666, 2.10545152,
#                    2.05666888, 2.39964692, 3.2517901, 2.7891825, 2.88763829,
#                    1.33773656, 3.2009672, 1.77276445]

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
fifteen_NN_result = []
for point in all_points:
    fifteen_NN_result.append((point, tree.kNN(point, k=5, dist=normalized_dist)[0]))
    # fifteen_NN_result.append((point, tree.kNN(point, k=5)))
end_t = datetime.datetime.now() - start_t

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
# for i, j in zip(np.concatenate((grief_heights, agony_heights, despair_heights)),
#                 np.concatenate((grief_weights, agony_weights, despair_weights))):
#     plt.annotate(s=("%.2f %.2f" % (i, j)), xy=(i, j))
plt.scatter([x[0] for x in fifteen_NN_yellow], [x[1] for x in fifteen_NN_yellow], s=50, c='yellow', marker='1',
            alpha=0.8)
plt.scatter([x[0] for x in fifteen_NN_blue], [x[1] for x in fifteen_NN_blue], s=50, c='blue', marker='2', alpha=0.8)
plt.scatter([x[0] for x in fifteen_NN_green], [x[1] for x in fifteen_NN_green], s=50, c='green', marker='3',
            alpha=0.8)

plt.axis((10, 70, 1, 7))
plt.title('5NN分类', size=30)
plt.xlabel('身高(cm)', size=15)
plt.ylabel('体重(kg)', size=15)

plt.show()
