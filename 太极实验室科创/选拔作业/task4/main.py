from B_enchmarkFunctions import BenchmarkFunctions
from R_UN import RUN
import matplotlib.pyplot as plt

if True:#基础设置可修改
    nP=50
    Func_name="F1"
    MaxIt=500

if True:#计算
    (lb,ub,dim,fobj)=BenchmarkFunctions(Func_name)
    (Best_fitness,BestPositions,Convergence_curve) = RUN(nP,MaxIt,lb,ub,dim,fobj)

if True:#绘图
    plt.figure()
    plt.semilogy(Convergence_curve[:-1], "r", linewidth=1)
    # plt.xlim(0,500)
    # plt.ylim(min(Convergence_curve),max(Convergence_curve))
    plt.box(True)
    plt.tight_layout()
    plt.show()