import numpy as np
import pandas as pd
import struct
import os
from pathlib import Path
from scipy import stats
from fitter import Fitter
import math


index=15
dat="dat_2000.{}".format(index)
location="location.{}".format(index)
Velocity="velocity.{}".format(index)
i_d="id.{}".format(index)
position="position_{}.csv".format(index)
velocity="velocity_{}.csv".format(index)
Id="id_{}.csv".format(index)
final="all_{}.csv".format(index)


Source_File = r"C:\in_Desktop\data\csst\source\{}".format(dat)
Cut_File    = r"C:\in_Desktop\data\csst\middle\{}".format(location)
Cut_File_v  = r"C:\in_Desktop\data\csst\middle\{}".format(Velocity)
Cut_File_id = r"C:\in_Desktop\data\csst\middle\{}".format(i_d)
Temporary_Files=r"C:\in_Desktop\data\csst\output\position\{}".format(position)
Temporary_Files_v=r"C:\in_Desktop\data\csst\output\velocity\{}".format(velocity)
Temporary_Files_id=r"C:\in_Desktop\data\csst\output\id\{}".format(Id)
Final_Files=r"C:\in_Desktop\data\csst\output\all\{}".format(final)



print("任务开始")

Source_File_path = Path(Source_File)   
size = Source_File_path.stat().st_size 
if size%32==0:
    particle_number=size//32
    print("粒子数为{}".format(particle_number))
else:
    print(particle_number=size//32)
particle_numbers=3*particle_number

print("开始读取坐标数据")
with open(Cut_File, "rb") as f:
    data = f.read(particle_numbers * 4)
    numbers = struct.unpack(f"{particle_numbers}i", data)
    numbers = list(numbers)
    Numbers = numbers
    # Numbers = numbers[0:9000]
print("坐标数据已读取")
print("开始转置坐标数据")
data_x = np.array(Numbers).reshape(-1, 3)
print("坐标数据已转置")

print("开始读取速度数据")
with open(Cut_File_v, "rb") as f: 
    data = f.read(particle_numbers * 4)
    numbers = struct.unpack(f"{particle_numbers}f", data)
    numbers = list(numbers)
    Numbers = numbers
print("速度数据已读取")
print("开始转置速度数据")
data_v = np.array(Numbers).reshape(-1, 3)
print("速度数据已转置")

print("开始读取id数据")
with open(Cut_File_id, "rb") as f: 
    data = f.read(particle_number * 8)
    numbers = struct.unpack(f"{particle_number}q", data)
    numbers = list(numbers)
    Numbers = numbers
print("id数据已读取")
print("开始放置id数据")
data_id = np.array(Numbers).reshape(-1, 1)
print("id数据已放置")

print("开始剪裁数据")
data_x=data_x[0:]
data_v=data_v[0:]
data_id=data_id[0:]

print("抽取拟合数据")
den_data= data_x[99::2000]#1000的倍数

print("开始简化数据")
Data_x=data_x[9::200]#100的倍数
Data_v=data_v[9::200]
Data_id=data_id[9::200]

print("开始进行拟合")
kde = stats.gaussian_kde(den_data.T)
print("拟合已完成")

print("开始进行计算密度值")
data_den=kde.evaluate(Data_x.T)*(10**23)
# data_den=kde.evaluate(Data_x[0])
# for i_den in range (1,1000000):
#     data_den0=kde.evaluate(Data_x[i_den])
#     data_den=np.hstack((data_den, data_den0))
#     print("计算完第{}个粒子位置的密度值".format(i_den))
# data_den = -np.log(data_den)*10
print("密度值计算完毕")

print("开始进行约化")
Data_den = data_den
# Data_den = np.around(data_den)#四舍五入
print("约化完毕")

print("开始进行数据合并")
# data = np.column_stack((Data_x, Data_v,Data_id,Data_den))
data = np.column_stack((Data_x,Data_den))
print("数据合并完毕")

# max_data=max(data[:, 3])
# min_data=min(data[:, 3])
max_data=max(data[:, 3])+1
min_data=min(data[:, 3])-1
multiple=100#设置组
proportion=(max_data-min_data)/multiple
print("最大的是{}，最小的是{}".format(max_data,min_data))
for i in range(0,multiple+1):
    data_cut=data[(data[:, 3] > min_data+i*proportion)& (data[:, 3] <= min_data+(i+1)*proportion)]
    density="cut{}_{}.csv".format(index,i)
    Density=r"C:\in_Desktop\data\csst\output\density\{}".format(density)
    df = pd.DataFrame(data_cut)
    df.to_csv(Density, index=False)
# data=data[(data[:, 3] > 0) & (data[:, 3] < 100)]
# data=data[(data[:, 3] > 10) ]



df = pd.DataFrame(data)
df.to_csv(Final_Files, index=False)



print("任务结束")









