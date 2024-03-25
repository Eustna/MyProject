import numpy as np
import pandas as pd
import struct
import os
from pathlib import Path
Source_File = r"C:\in_Desktop\data\input\dat_2000.1"
Cut_File    = r"C:\in_Desktop\data\try\dat_2000_location.1"
Temporary_Files=r"C:\in_Desktop\data\try\locatioz.csv"

p = Path(Source_File)   
size = p.stat().st_size 
if size%32==0:
    i=size//32
    print("粒子数为{}".format(i))
else:
    print(i=size//32)

with open(Source_File, 'rb') as f1:     
    data = f1.read(12*i)                
    with open(Cut_File, 'wb') as f2:    
        f2.write(data)   
j=3*i
with open(Cut_File, "rb") as f:
    data = f.read(j * 4)
    numbers = struct.unpack(f"{j}i", data)
    numbers = list(numbers)
    Numbers = numbers[0:9000]


# 将列表转换为 numpy 数组并重塑为每行三个数据
data = np.array(Numbers).reshape(-1, 3)
first_column = data[:, 2]#提取第一列
data=first_column

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 写入 CSV 文件
df.to_csv(Temporary_Files, index=False)