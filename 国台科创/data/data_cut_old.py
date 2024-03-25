import struct
import pandas as pd
import os
from pathlib import Path
Source_File = r"C:\in_Desktop\data\input\dat_2000.1"
Cut_File    = r"C:\in_Desktop\data\output\coordinate\dat_2000_location.1"
Cut_File_v  = r"C:\in_Desktop\data\output\coordinate\dat_2000_velocity.1"
Cut_File_id = r"C:\in_Desktop\data\output\coordinate\dat_2000_id.1"
Temporary_Files=r"C:\in_Desktop\data\output\coordinate_csv\dat_2000_location_1_snap.csv"
Temporary_Files_v=r"C:\in_Desktop\data\output\coordinate_csv\dat_2000_location_1_snap_v.csv"
Temporary_Files_id=r"C:\in_Desktop\data\output\coordinate_csv\dat_2000_location_1_snap_id.csv"
Final_Files = r"C:\in_Desktop\data\output\coordinate_csv\dat_2000_location_1.csv"
print("开始任务")

p = Path(Source_File)   
size = p.stat().st_size 
if size%32==0:
    i=size//32
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
        data = f.read(i * 4)
        numbers = struct.unpack(f"{i}i", data)
        numbers = list(numbers)
        df = pd.DataFrame(numbers)
        df.to_csv(Temporary_Files, index=False)
    print("已完成坐标数据转换")

print("开始速度数据转换")
if os.path.exists(Temporary_Files_v):
    print("找到速度数据转换文件")
else:
    with open(Cut_File_v, "rb") as f: 
        data = f.read(i * 4)
        numbers = struct.unpack(f"{i}f", data)
        numbers = list(numbers)
        df = pd.DataFrame(numbers)
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
        df = pd.DataFrame(numbers)
        df.to_csv(Temporary_Files_id, index=False)
    print("已完成id数据转换")

print("开始数据转置")
dfc = pd.read_csv(Temporary_Files, header=None)
dfv = pd.read_csv(Temporary_Files_v, header=None)
dfid = pd.read_csv(Temporary_Files_id, header=None)
if os.path.exists(Final_Files):
    df = pd.read_csv(Final_Files, header=None)
    ct = int(df.loc[0,3])
    for x in range(ct,i):
        df.loc[x,0] = dfc.iloc[3*x+1, 0]
        df.loc[x,1] = dfc.iloc[3*x+2, 0]
        df.loc[x,2] = dfc.iloc[3*x+3, 0]
        df.loc[0,3] = x
        df.loc[x,4] = dfv.iloc[3*x+1, 0]
        df.loc[x,5] = dfv.iloc[3*x+2, 0]
        df.loc[x,6] = dfv.iloc[3*x+3, 0]
        df.loc[x,7] = dfid.iloc[x+1,0]
        print("已完成粒子{}的数据转置,还剩{}个".format(x+1,i-x))
        if x%100000==0:
            df.to_csv(Final_Files, header=None, index=None)
    df.to_csv(Final_Files, header=None, index=None)
else:
    df = pd.DataFrame()
    for x in range(0,i):
        df.loc[x,0] = dfc.iloc[3*x+1, 0]
        df.loc[x,1] = dfc.iloc[3*x+2, 0]
        df.loc[x,2] = dfc.iloc[3*x+3, 0]
        df.loc[0,3] = x
        df.loc[x,4] = dfv.iloc[3*x+1, 0]
        df.loc[x,5] = dfv.iloc[3*x+2, 0]
        df.loc[x,6] = dfv.iloc[3*x+3, 0]
        df.loc[x,7] = dfid.iloc[x+1,0]
        print("已完成粒子{}的数据转置,还剩{}个".format(x+1,i-x))
        if x%100000==0:
            df.to_csv(Final_Files, header=None, index=None)
    df.to_csv(Final_Files, header=None, index=None)


print("删除多余文件")
try:
    os.remove(Temporary_Files)
    print(f'文件 {Temporary_Files} 已被删除')
except FileNotFoundError:
    print(f'文件 {Temporary_Files} 不存在')
try:
    os.remove(Temporary_Files_v)
    print(f'文件 {Temporary_Files_v} 已被删除')
except FileNotFoundError:
    print(f'文件 {Temporary_Files_v} 不存在')
try:
    os.remove(Temporary_Files_id)
    print(f'文件 {Temporary_Files_id} 已被删除')
except FileNotFoundError:
    print(f'文件 {Temporary_Files_id} 不存在')


print("任务结束")