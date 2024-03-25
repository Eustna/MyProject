import numpy as np
import pandas as pd
import struct
import os
from pathlib import Path
from scipy import stats
from fitter import Fitter
import math


Source_File = r"C:\in_Desktop\data\input\dat_2000.1"
Cut_File    = r"C:\in_Desktop\data\try\dat_2000_location.1"
Temporary_Files=r"C:\in_Desktop\data\try\density1.csv"

Source_File_path = Path(Source_File)   
size = Source_File_path.stat().st_size 
if size%32==0:
    particle_number=size//32
    print("粒子数为{}".format(particle_number))
else:
    print(particle_number=size//32)

with open(Source_File, 'rb') as f1:     
    data = f1.read(12*particle_number)                
    with open(Cut_File, 'wb') as f2:    
        f2.write(data)   
particle_numbers=3*particle_number
with open(Cut_File, "rb") as f:
    data = f.read(particle_numbers * 4)
    numbers = struct.unpack(f"{particle_numbers}i", data)
    numbers = list(numbers)
    Numbers = numbers[0:9000]

data = np.array(Numbers).reshape(-1, 3)
# x= data[:, 0]#提取第1列
# y= data[:, 1]#提取第2列
# z= data[:, 2]#提取第3列


# 使用 scipy 的 stats 模块来拟合你的数据
kde = stats.gaussian_kde(data.T)

# # 生成一个网格来评估密度
# x_grid = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
# y_grid = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
# z_grid = np.linspace(min(data[:, 2]), max(data[:, 2]), 100)
# X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

# # 计算网格上每一点的密度
# density = kde(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]))

# # 将密度转换为一个与网格相同形状的数组
# density = density.reshape(X.shape)
result=kde.evaluate(data.T)*10**17

result = np.around(result)#四舍五入

data = np.column_stack((data, result))
# data=data[(data[:, 3] > 0) & (data[:, 3] < 100)]
data=data[(data[:, 3] > 10) ]
# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 写入 CSV 文件
df.to_csv(Temporary_Files, index=False)





















# from scipy import stats

# # 假设你有一个二维数组 data，其中每一行代表一个粒子的坐标
# data = np.array([x, y, z]).T

# # 使用 scipy 的 gaussian_kde 函数来拟合你的数据
# kde = stats.gaussian_kde(data)

# # 现在，你可以使用 kde 函数来计算任何点 (x, y, z) 的密度
# density = kde.evaluate([x_point, y_point, z_point])








# import numpy as np

# # 假设我们有两个二维数组
# array1 = np.array([[1, 2], [3, 4]])
# array2 = np.array([[5, 6], [7, 8]])

# # 将 array2 接入 array1 的列末尾
# result = np.hstack((array1, array2))

# print(result)