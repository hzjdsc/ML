import operator


class KdTree(object):

    # point_list是一个list的point, point[0]是一个tuple的特征，point[1]是类别
    def __init__(self, point_list, depth=0, root=None):

        if len(point_list) > 0:
            # 轮换按照树深度选择坐标轴, k为特征向量的维度
            k = len(point_list[0][0])
            axis = depth % k

            # 选中位线，切
            point_list.sort(key=lambda x: x[0][axis])
            median = len(point_list) // 2

            self.axis = axis
            self.root = root
            self.size = len(point_list)

            # 造节点
            self.node = point_list[median]
            # 递归造左枝和右枝
            if len(point_list[:median]) > 0:
                self.left = KdTree(point_list[:median], depth + 1, self)
            else:
                self.left = None
            if len(point_list[median + 1:]) > 0:
                self.right = KdTree(point_list[median + 1:], depth + 1, self)
            else:
                self.right = None
        else:
            return None

    def find_leaf(self, point):
        if self.left == None and self.right == None:
            return self
        elif self.left == None:
            return self.right.find_leaf(point)
        elif self.right == None:
            return self.left.find_leaf(point)
        elif point[self.axis] < self.node[0][self.axis]:
            return self.left.find_leaf(point)
        else:
            return self.right.find_leaf(point)

    # 查找最近的k个点，复杂度O(DlogN), D是维度，N是树的大小
    # 输入 点，k，距离函数（默认是L2）
    def knearest(self, point, k=1, dist=lambda x, y: sum(map(lambda u, v: (u - v) ** 2, x, y))):
        # 往下戳到最底也
        leaf = self.find_leaf(point)
        # 从叶子往上爬
        return leaf.k_down_up(point, k, dist, result=[], stop=self, visited=None)

    # 从下往上爬函数，stop是到哪里去，visited是从哪里来
    def k_down_up(self, point, k, dist, result=[], stop=None, visited=None):
        # 选最长距离
        if result == []:
            max_dist = 0
        else:
            max_dist = max([x[1] for x in result])
        other_result = []

        # 如果离分界线的距离小于现在的最大距离，或者数据点不够，就从另一边的树根开始刨
        if (self.left == visited and self.node[0][self.axis] - point[self.axis] < max_dist and self.right != None) \
                or (len(result) < k and self.left == visited and self.right != None):
            other_result = self.right.knearest(point, k, dist)

        if (self.right == visited and point[self.axis] - self.node[0][self.axis] < max_dist and self.left != None) \
                or (len(result) < k and self.right == visited and self.left != None):
            other_result = self.left.knearest(point, k, dist)

        # 刨出来的点放在一起，选前k个
        result.append((self.node, dist(point, self.node[0])))
        result = sorted(result + other_result, key=lambda pair: pair[1])[:k]

        # 到停点就返回结果
        if self == stop:
            return result
        # 没有就带着现有结果接着往上爬
        else:
            return self.root.k_down_up(point, k, dist, result, stop, self)

    # 输入 特征、类别、k、距离函数
    # 返回这个点属于该类别的概率
    def kNN_prob(self, point, label, k, dist=lambda x, y: sum(map(lambda u, v: (u - v) ** 2, x, y))):
        nearests = self.knearest(point, k, dist)
        return float(len([pair for pair in nearests if pair[0][1] == label])) / float(len(nearests))

    # 输入 特征、k、距离函数
    # 返回该点概率最大的类别以及相对应的概率
    def kNN(self, point, k, dist=lambda x, y: sum(map(lambda u, v: (u - v) ** 2, x, y))):
        nearests = self.knearest(point, k, dist)
        statistics = {}
        for data in nearests:
            label = data[0][1]
            if label not in statistics:
                statistics[label] = 1
            else:
                statistics[label] += 1
        max_label = max(statistics.items(), key=operator.itemgetter(1))[0]
        return max_label, float(statistics[max_label]) / float(len(nearests))

#
# if __name__ == '__main__':
#     import numpy as np
#
#     x = [1,1,2,3,3]
#     y = [1,2,2,1,2]
#     l = ['0','1','2','3','4']
#
#     grieves = map(lambda x,y,z: tuple(((x,y),z)), x, y, l)
#     tree = KdTree(list(grieves))
#     point = [2, 1.6]
#     print(tree.kNN(point, 2))k
