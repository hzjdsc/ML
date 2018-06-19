# from src.KdTree import KdTree
# import numpy as np

# grief_heights = np.random.normal(50, 6, 3)
# grief_weights = np.random.normal(5, 0.5, 3)
#
# grieves = map(lambda x,y: tuple(((x,y),'g')), grief_heights, grief_weights)
# tree = KdTree(list(grieves))
# print(tree)
result = [(1,5),(4,3)]
m = max([x[1] for x in result])
print(m)