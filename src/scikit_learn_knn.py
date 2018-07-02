from sklearn import neighbors
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 7

x1 = np.random.normal(50, 6, 100)
y1 = np.random.normal(5, 0.5, 100)

x2 = np.random.normal(30, 6, 100)
y2 = np.random.normal(4, 0.5, 100)

x3 = np.random.normal(45, 6, 100)
y3 = np.random.normal(2.5, 0.5, 100)

x_val = np.concatenate((x1, x2, x3))
y_val = np.concatenate((y1, y2, y3))

x_diff = max(x_val) - min(x_val)
y_diff = max(y_val) - min(y_val)

x_normalized = x_val / x_diff
y_normalized = y_val / y_diff
xy_normalized = list(zip(x_normalized, y_normalized))

labels = [1] * 100 + [2] * 100 + [3] * 100

clf = neighbors.KNeighborsClassifier(5)
clf.fit(xy_normalized, labels)


# 测试点（30，3）的情况
# nearest = clf.kneighbors([(30/x_diff, 3/y_diff)], 5, False)
# plt.scatter(30, 3, c='y', s=200, marker='*', alpha=0.8, zorder=10)
# for index in nearest[0]:
#     plt.scatter(x_diff*xy_normalized[index][0], y_diff*xy_normalized[index][1], c='y', s=50, marker='1', alpha=0.8, zorder=10)
#
# plt.scatter(x1, y1, c='b', marker='s', s=50, alpha=0.8)
# plt.scatter(x2, y2, c='r', marker='^', s=50, alpha=0.8)
# plt.scatter(x3, y3, c='g', s=50, alpha=0.8)
#
# prediction = clf.predict([(30/x_diff, 3/y_diff)])
# prediction_proba = clf.predict_proba([(30/x_diff, 3/y_diff)])
# print(prediction)
# print(prediction_proba)


# 模型打分
# x1_test = np.random.normal(50, 6, 100)
# y1_test = np.random.normal(5, 0.5, 100)
#
# x2_test = np.random.normal(30, 6, 100)
# y2_test = np.random.normal(4, 0.5, 100)
#
# x3_test = np.random.normal(45, 6, 100)
# y3_test = np.random.normal(2.5, 0.5, 100)
# xy_test_normalized = list(zip(np.concatenate((x1_test, x2_test, x3_test)) / x_diff,
#                               np.concatenate((y1_test, y2_test, y3_test)) / y_diff))
#
# labels_test = [1] * 100 + [2] * 100 + [3] * 100
# score = clf.score(xy_test_normalized, labels_test)
# clf2 = neighbors.KNeighborsClassifier(1)
# clf2.fit(xy_normalized, labels)
# score2 = clf2.score(xy_test_normalized, labels_test)
# print(score)
# print(score2)


#画图
xx,yy = np.meshgrid(np.arange(1,70.1,1), np.arange(1,7.01,0.1))
xx_normalized = xx/x_diff
yy_normalized = yy/y_diff
coords = np.c_[xx_normalized.ravel(), yy_normalized.ravel()]
z = clf.predict(coords)
z=z.reshape(xx.shape)
light_rgb = ListedColormap(['#AAAAFF','#FFAAAA','#AAFFAA'])

z_proba = clf.predict_proba(coords)
z_proba_reds = z_proba[:,1].reshape(xx.shape)

# plt.pcolormesh(xx, yy, z, cmap=light_rgb)
plt.pcolormesh(xx, yy, z_proba_reds, cmap='Reds')
plt.scatter(x1, y1, c='b', marker='s', s=50, alpha=0.8)
plt.scatter(x2, y2, c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3, y3, c='g', s=50, alpha=0.8)
plt.axis((10, 70, 1, 7))


plt.show()
