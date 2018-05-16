import numpy as np
from sklearn.neighbors import KDTree

np.random.seed(0)
X=np.random.random((10,3))
tree=KDTree(X,leaf_size=3)
dist, ind=tree.query([X[0]],k=3)
print(ind)
print(dist)