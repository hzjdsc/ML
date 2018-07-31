from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO


# load_iris是测试数据
iris = load_iris()

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(iris.data, iris.target)

sample_idx = 111
prediction = clf.predict(iris.data[sample_idx:sample_idx+1])
truth = iris.target[sample_idx]
# print(prediction, truth)

class_probabilities = clf.predict_proba(iris.data[sample_idx:sample_idx+1])
# print(class_probabilities)

# 用于展示不同因子的重要性
# print(clf.feature_importances_)

# 创建缓存变量
f = StringIO()
# 把决策树clf的图形结果输出，存入缓存中
tree.export_graphviz(clf, out_file=f)
graph = pydotplus.graph_from_dot_data(f.get_value())
# 将图片保存进本地文件中
graph.write_png("dtree2.png")
# 画出决策树，也可以用标注的代码



