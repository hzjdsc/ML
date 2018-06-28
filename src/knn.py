import random
import operator
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.KdTree import KdTree

# 100个正态分布的悲伤
# grief_heights = np.random.normal(50, 6, 20)
# grief_weights = np.random.normal(5, 0.5, 20)
grief_heights = [52.06105112, 45.29148154, 53.09354958, 46.64303881, 42.44441465, 47.28922841, 49.24488817, 43.80745517,
                 54.60271404, 52.99178818, 58.8238066, 39.31671937, 47.19095059, 46.06672843, 50.46985025, 46.6073706,
                 42.96313143, 50.06570039, 52.77657516, 49.31967616]
grief_weights = [4.64533944, 5.13382608, 5.14936688, 4.94655363, 3.83105107, 5.48957463, 3.27899664, 5.38285074,
                 4.43649426, 4.80306774, 6.56714855, 5.72951018, 4.79345601, 5.44538042, 5.0999316, 5.26353153,
                 4.20818748, 5.15959703, 4.94603446, 5.56005541]

# 100个正态分布的痛苦
# agony_heights = np.random.normal(30, 6, 20)
# agony_weights = np.random.normal(4, 0.5, 20)
agony_heights = [24.51113812, 22.9572049, 36.18256777, 17.97887871, 29.15955975, 27.79304777, 31.45053713, 28.51423368,
                 29.80585883, 27.76625555, 19.5845069, 33.3832776, 32.5143978, 19.20623492, 25.98263527, 30.39058502,
                 27.82357139, 30.14269055, 12.89093926, 29.95304237]
agony_weights = [5.18343006, 4.74482892, 4.63984167, 3.71421141, 3.5018431, 3.53049261, 4.19781959, 3.61778993,
                 2.8609095, 4.3211517, 4.21329843, 4.0895797, 4.16511166, 4.90906785, 4.79833652, 4.2042287, 4.12361779,
                 4.00871815, 4.34458511, 4.65541531]

# 100个正态分布的绝望
# despair_heights = np.random.normal(45, 6, 20)
# despair_weights = np.random.normal(2.5, 0.5, 20)
despair_heights = [47.60772409, 41.0401255, 49.97170585, 43.16310204, 49.41393005, 40.53085895, 28.21783366,
                   49.78131412, 44.50355543, 51.92690906, 31.55062205, 49.38882281, 45.18332532, 45.35056474,
                   32.11078176, 36.99199215, 41.01105232, 55.65058493, 39.79352219, 55.7779951]
despair_weights = [2.8144508, 2.83478508, 2.23171516, 1.9424559, 2.43355543, 3.59951873, 1.63326666, 2.10545152,
                   2.05666888, 3.21161305, 2.06854284, 2.85709833, 2.39964692, 3.2517901, 2.7891825, 2.88763829,
                   2.22755321, 1.33773656, 3.2009672, 1.77276445]

plt.rcParams['font.sans-serif'] = ['SimHei']
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10

# 设置样本集
grieves = map(lambda x, y: tuple(((x, y), 'g')), grief_heights, grief_weights)
agonies = map(lambda x, y: tuple(((x, y), 'b')), grief_heights, grief_weights)
despairs = map(lambda x, y: tuple(((x, y), 'y')), grief_heights, grief_weights)

# 创建kd树
tree = KdTree(list(grieves) + list(agonies) + list(despairs))


# 穷举生成空间上的点
# all_points = []
# for i in range(100, 701, 30):
#     for j in range(100, 701, 30):
#         all_points.append((float(i) / 10., float(j) / 100.))
#
# print(len(all_points))


# 设置归一化距离函数
def normalized_dist(x, y):
    return (x[0] - y[0]) ** 2 + (10 * x[1] - 10 * y[1]) ** 2


# # 每个点运算15NN，并记录计算时间
# start_t = datetime.datetime.now()
# fifteen_NN_result = []
# for point in all_points:
#     fifteen_NN_result.append((point, tree.kNN(point, k=5, dist=normalized_dist)[0]))
# end_t = datetime.datetime.now() - start_t
#
# # 把每个颜色的数据分开
# fifteen_NN_yellow = []
# fifteen_NN_green = []
# fifteen_NN_blue = []
#
# for pair in fifteen_NN_result:
#     if pair[1] == 'y':
#         fifteen_NN_yellow.append(pair[0])
#     if pair[1] == 'g':
#         fifteen_NN_green.append(pair[0])
#     if pair[1] == 'b':
#         fifteen_NN_blue.append(pair[0])

plt.scatter(40, 2.7, c='r', s=200, marker='*', alpha=0.8, zorder=10)
print((tree.kNN([40, 2.7], k=3, dist=normalized_dist)[0]))
plt.scatter(grief_heights, grief_weights, c='g', marker='s', s=50, alpha=0.8, zorder=10)
plt.scatter(agony_heights, agony_weights, c='b', marker='^', s=50, alpha=0.8, zorder=10)
plt.scatter(despair_heights, despair_weights, c='y', s=50, alpha=0.8, zorder=10)
for i, j in zip(grief_heights + agony_heights + despair_heights, grief_weights + agony_weights + despair_weights):
    plt.annotate(s=("%.2f %.2f" % (i, j)), xy=(i, j))
# plt.scatter([x[0] for x in fifteen_NN_yellow], [x[1] for x in fifteen_NN_yellow], s=50, c='yellow', marker='1',
#             alpha=0.8)
# plt.scatter([x[0] for x in fifteen_NN_blue], [x[1] for x in fifteen_NN_blue], s=50, c='blue', marker='2', alpha=0.8)
# plt.scatter([x[0] for x in fifteen_NN_green],[x[1] for x in fifteen_NN_green], s=50, c='green', marker='3', alpha=0.8)
plt.axis((10, 70, 1, 7))
plt.title('5NN分类', size=30)
plt.xlabel('身高(cm)', size=15)
plt.ylabel('体重(kg)', size=15)

plt.show()
