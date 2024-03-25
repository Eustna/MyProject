# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # 创建一个图形对象
# fig = plt.figure()

# # 创建一个3D坐标系
# ax = fig.add_subplot(111, projection='3d')

# # 生成一些随机数据点
# x_vals = np.random.randn(100)
# y_vals = np.random.randn(100)
# z_vals = np.random.randn(100)

# # 在3D坐标系中绘制数据点
# ax.scatter(x_vals, y_vals, z_vals)

# # 显示图形
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata

# # 假设你已经有了一些散点数据
# x = np.random.rand(100)
# y = np.random.rand(100)
# z = np.random.rand(100)




# # 将离散数据点拟合成曲面
# points = np.column_stack((x, y))
# values = z

# # 定义拟合网格
# xi = np.linspace(min(x), max(x), 100)
# yi = np.linspace(min(y), max(y), 100)
# XI, YI = np.meshgrid(xi, yi)

# # 进行曲面拟合
# ZI = griddata(points, values, (XI, YI), method='cubic')

# # 创建一个图形对象
# fig = plt.figure()

# # 创建一个3D坐标系
# ax = fig.add_subplot(111, projection='3d')

# # 在3D坐标系中绘制拟合曲面
# ax.plot_surface(XI, YI, ZI)
# # # 在3D坐标系中绘制数据点
# ax.scatter(x, y, z)

# # 显示图形
# plt.show()













# import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 假设你已经有了一些散点数据
# points = np.random.rand(100, 3)   # 30个随机点

# # 计算凸包
# hull = ConvexHull(points)

# # 创建一个图形对象
# fig = plt.figure()

# # 创建一个3D坐标系
# ax = fig.add_subplot(111, projection='3d')

# # 绘制原始点
# ax.scatter(points[:,0], points[:,1], points[:,2])

# # 绘制凸包的边界
# for s in hull.simplices:
#     s = np.append(s, s[0])  # 在这里循环以获取闭合的多边形
#     ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")

# # 显示图形
# plt.show()










# import numpy as np
# from scipy.spatial import ConvexHull
# from sklearn.set import DBSCAN
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 假设你已经有了一些散点数据
# points = np.random.rand(1000, 3)   # 100个随机点

# # 使用DBSCAN进行聚类
# seting = DBSCAN(eps=0.5, min_samples=2).fit(points)
# labels = seting.labels_

# # 创建一个图形对象
# fig = plt.figure()

# # 创建一个3D坐标系
# ax = fig.add_subplot(111, projection='3d')

# # 对每个聚类进行凸包计算并绘制
# for i in np.unique(labels):
#     if i == -1:
#         continue  # 忽略噪声点
#     set_points = points[labels == i]
#     hull = ConvexHull(set_points)
#     for s in hull.simplices:
#         s = np.append(s, s[0])  # 在这里循环以获取闭合的多边形
#         ax.plot(set_points[s, 0], set_points[s, 1], set_points[s, 2], "r-")
# ax.scatter(points[:,0], points[:,1], points[:,2])
# # 显示图形
# plt.show()







import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import time
time_now = int(time.time())

np.random.seed(time_now)

# 初始化一个空的列表来存储所有的聚集
sets = []

# 生成10到20个聚集
for _ in range(np.random.randint(10, 21)):
    # 为每个聚集生成一个随机的均值
    mean = np.random.rand(3) * 20  # 在0到10之间随机选择

    # 我们将协方差矩阵设为单位矩阵
    cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # 生成聚集的点
    set = np.random.multivariate_normal(mean, cov, 100)

    # 将聚集的点添加到列表中
    sets.append(set)

# 将所有的聚集合并成一个数组
points = np.concatenate(sets)


# 使用DBSCAN进行聚类
seting = DBSCAN(eps=0.8, min_samples=2).fit(points)
labels = seting.labels_

# 创建一个新的图形对象
fig = plt.figure()

# 创建一个新的3D坐标系
ax = fig.add_subplot(111, projection='3d')

# 对每个聚类进行凸包计算并绘制
for i in np.unique(labels):
    if i == -1:
        continue  # 忽略噪声点
    set_points = points[labels == i]
    if len(set_points) < 4:  # 对于三维数据，至少需要4个点
        continue
    hull = ConvexHull(set_points)
    for s in hull.simplices:
        s = np.append(s, s[0])  # 在这里循环以获取闭合的多边形
        ax.plot(set_points[s, 0], set_points[s, 1], set_points[s, 2], "r-")
# ax.scatter(points[:,0], points[:,1], points[:,2])
# 显示图形
plt.show()















# # 创建一个图形对象
# fig = plt.figure()

# # 创建一个3D坐标系
# ax = fig.add_subplot(111, projection='3d')

# # 在3D坐标系中绘制点
# ax.scatter(points[:,0], points[:,1], points[:,2])

# # 显示图形
# plt.show()










