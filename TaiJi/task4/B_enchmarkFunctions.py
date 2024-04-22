import numpy as np

def BenchmarkFunctions(F):

    def F1(x):
        D = len(x)
        z = x[0]**2 + 10**6 * np.sum(x[1:D]**2)#看情况改
        return float(z)

    def F2(x):
        D = len(x)
        z = 0
        for i in range(D):
            z += abs(x[i])**(i+2)
        return float(z)
    
    def F3(x):
        x = np.array(x)
        z = np.sum(x**2) + (np.sum(0.5*x))**2 + (np.sum(0.5*x))**4
        return float(z)
    
    def F4(x):
        D = len(x)
        z = 0
        for i in range(D-1):
            z += 100*(x[i]**2-x[i+1]**2+(x[i]-1)**2)
        return float(z)
    
    def F5(x):
        D = len(x)
        z = 10**6*x[0]+np.sum(x[1:D]**2)
        return float(z)



    # def Ufun(x,a,k,m):
    #     o = k * np.where(x > a, (x - a) ** m, 0) + k * np.where(x < -a, (-x - a) ** m, 0)
    #     return o

    D = 30

    if F == "F1":
        fobj = F1
        lb=-100
        ub=100
        dim=D
    elif F == "F2":
        fobj = F2
        lb=-100
        ub=100
        dim=D
    elif F == "F3":
        fobj = F3
        lb=-100
        ub=100
        dim=D
    elif F == "F4":
        fobj = F4
        lb=-100
        ub=100
        dim=D
    elif F == "F5":
        fobj = F5
        lb=-100
        ub=100
        dim=D
    
    return lb,ub,dim,fobj

# A=BenchmarkFunctions("F1")
