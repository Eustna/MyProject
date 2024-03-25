import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import time
time_now = int(time.time())
np.random.seed(time_now)
sets = []
for _ in range(np.random.randint(10, 21)):
    mean = np.random.rand(3) * 20
    cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    set = np.random.multivariate_normal(mean, cov, 100)
    sets.append(set)
points = np.concatenate(sets)
seting = DBSCAN(eps=0.8, min_samples=2).fit(points)#可修改eps值来确定成团的尺度大小
labels = seting.labels_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.unique(labels):
    if i == -1:
        continue
    set_points = points[labels == i]
    if len(set_points) < 4:
        continue
    hull = ConvexHull(set_points)
    for s in hull.simplices:
        s = np.append(s, s[0])
        ax.plot(set_points[s, 0], set_points[s, 1], set_points[s, 2], "r-")
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()