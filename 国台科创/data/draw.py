import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print("开始")
resource = r"C:\in_Desktop\data\output\coordinate_csv\dat_2000_location_1.csv"
df = pd.read_csv(resource, nrows=300000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],s=0.0005)
plt.show()
print("结束")