import struct
import pandas as pd
import os
from pathlib import Path
import numpy as np
import struct

index=15
dat="dat_2000.{}".format(index)
location="location.{}".format(index)
Velocity="velocity.{}".format(index)
i_d="id.{}".format(index)
position="position_{}.csv".format(index)
velocity="velocity_{}.csv".format(index)
Id="id_{}.csv".format(index)

Source_File = r"C:\in_Desktop\data\csst\source\{}".format(dat)
Cut_File    = r"C:\in_Desktop\data\csst\middle\{}".format(location)
Cut_File_v  = r"C:\in_Desktop\data\csst\middle\{}".format(Velocity)
Cut_File_id = r"C:\in_Desktop\data\csst\middle\{}".format(i_d)
Temporary_Files=r"C:\in_Desktop\data\csst\output\position\{}".format(position)
Temporary_Files_v=r"C:\in_Desktop\data\csst\output\velocity\{}".format(velocity)
Temporary_Files_id=r"C:\in_Desktop\data\csst\output\id\{}".format(Id)
print("开始任务")

p = Path(Source_File)   
size = p.stat().st_size 
if size%32==0:
    i=size//32
    j=i*3
    print("粒子数为{}".format(i))
else:
    print(i=size//32)

print("开始坐标数据切割")
if os.path.exists(Cut_File):
    print("找到坐标数据切割文件")
else:
    with open(Source_File, 'rb') as f1:     
        data = f1.read(12*i)                
        with open(Cut_File, 'wb') as f2:    
            f2.write(data)                  
    print("已完成坐标数据切割")


print("开始速度数据切割")
if os.path.exists(Cut_File_v):
    print("找到速度数据切割文件")
else:
    with open(Source_File, 'rb') as f1:
        f1.seek(12*i)
        data = f1.read(12*i) 
        with open(Cut_File_v, 'wb') as f2: 
            f2.write(data) 
    print("已完成速度数据切割")

print("开始id数据切割")
if os.path.exists(Cut_File_id):
    print("找到id数据切割文件")
else:
    with open(Source_File, 'rb') as f1: 
        f1.seek(24*i)
        data = f1.read() 
        with open(Cut_File_id, 'wb') as f2: 
            f2.write(data) 
    print("已完成id数据切割")


print("开始坐标数据转换")
if os.path.exists(Temporary_Files):
    print("找到坐标数据转换文件")
else:
    with open(Cut_File, "rb") as f:
        data = f.read(j * 4)
        numbers = struct.unpack(f"{j}i", data)
        numbers = list(numbers)
        data = np.array(numbers).reshape(-1, 3)
        df = pd.DataFrame(data)
        df.to_csv(Temporary_Files, index=False)
    print("已完成坐标数据转换")

print("开始速度数据转换")
if os.path.exists(Temporary_Files_v):
    print("找到速度数据转换文件")
else:
    with open(Cut_File_v, "rb") as f: 
        data = f.read(j * 4)
        numbers = struct.unpack(f"{j}f", data)
        numbers = list(numbers)
        data = np.array(numbers).reshape(-1, 3)
        df = pd.DataFrame(data)
        df.to_csv(Temporary_Files_v, index=False)
    print("已完成速度数据转换")

print("开始id数据转换")
if os.path.exists(Temporary_Files_id):
    print("找到id数据转换文件")
else:
    with open(Cut_File_id, "rb") as f: 
        data = f.read(i * 8)
        numbers = struct.unpack(f"{i}q", data)
        numbers = list(numbers)
        data = np.array(numbers).reshape(-1, 1)
        df = pd.DataFrame(data)
        df.to_csv(Temporary_Files_id, index=False)
    print("已完成id数据转换")
print("任务结束")