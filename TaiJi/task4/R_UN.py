import numpy as np

def standardization(x):
    x = np.array(x)
    shape = x.shape
    if len(shape) == 1:
        x = [x]
    elif shape[0] == 1:
        x = x.reshape(-1, 1)
    elif shape[1] == 1:
        pass
    return x

def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z

def RungeKutta(XB, XW, DelX):
        shape = np.array(XB).shape
        if len(shape) == 1:
            dim = 1
        elif shape[0] == 1:
            dim = XB.shape[1]
        elif shape[1] == 1:
            dim = XB.shape[0]
        
        C = np.random.randint(1, 3) * (1 - np.random.rand())
        r1 = np.random.rand(1, dim)
        r2 = np.random.rand(1, dim)

        K1 = 0.5 * (np.random.rand() * XW - C * XB)
        K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2) - (C * XB + r1 * K1 * DelX / 2))
        K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2) - (C * XB + r1 * K2 * DelX / 2))
        K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

        XRK = (K1 + 2 * K2 + 2 * K3 + K4)
        SM = 1/6 * XRK
        return SM
    
def RndX(nP, i):
    Qi = np.random.permutation(nP)
    Qi = [q for q in Qi if q != i]
    A, B, C = Qi[:3]
    return A, B, C

def initialization(nP, dim, ub, lb):
    if isinstance(ub, (int, float)):
        Boundary_no = 1
    else:
        Boundary_no = len(ub)
    if Boundary_no == 1:
        X = np.random.rand(nP, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        X = np.zeros((nP, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(nP) * (ub_i - lb_i) + lb_i
    return X


def RUN(nP,MaxIt,lb,ub,dim,fobj):
    Cost = np.zeros((nP, 1))
    X = initialization(nP, dim, ub, lb)

    Xnew2 = np.zeros((1, dim))
    Convergence_curve = np.zeros((1, MaxIt))

    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    Best_Cost = np.min(Cost)
    ind = np.argmin(Cost)
    Best_X = X[ind, :].reshape(1, -1)

    Convergence_curve[0] = Best_Cost

    it = 0

    while it < MaxIt:
        it += 1
        f = 20 * np.exp(-(12 * (it / MaxIt)))
        Xavg = np.mean(X,axis=0).reshape(1, -1)
        SF = (2 * (0.5 - np.random.rand(nP,1)) * f)

        for i in range(nP):
            ind_l = np.argmin(Cost)
            lBest = X[ind_l, :].reshape(1, -1)
            A, B, C = RndX(nP, i)
            ind1 = np.argmin(Cost[[A, B, C]])
            gama = (np.random.rand() * (X[i, :] - np.random.rand(dim)*(ub - lb))*np.exp( - 4 * it / MaxIt)).reshape(1, -1)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * (np.abs(Stp))
            if Cost[i] < Cost[ind1]:                
                Xb = X[i,:].reshape(1, -1)
                Xw = X[ind1,:].reshape(1, -1)
            else:
                Xb = X[ind1,:].reshape(1, -1)
                Xw = X[i,:].reshape(1, -1)
            
            SM = RungeKutta(Xb, Xw, DelX)
            L = (np.random.rand(dim) < 0.5).reshape(1, -1)
            Xc = L * X[i, :] + (1 - L) * X[A, :]
            Xm = L * Best_X + (1 - L) * lBest
            vec = [1, -1]
            flag = np.floor(2 * np.random.rand(dim) + 1).astype(int)

            r = np.zeros(dim)
            for j in range(dim):
                r[j] = vec[flag[j]-1]
            r = r.reshape(1, -1)

            g = 2 * np.random.rand()
            mu = (0.5 + .1 * np.random.randn(dim)).reshape(1, -1)

            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * (SM) + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * (SM) + mu * (X[A, :] - X[B, :])
            
            FU = Xnew > ub
            FL = Xnew < lb

            Xnew = ((Xnew * (~(FU + FL))) + ub * FU + lb * FL)
            CostNew = fobj(Xnew.reshape(-1, 1))

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                X[i] = CostNew

            if np.random.rand() < 0.5:

                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.floor(float(Unifrnd(-1, 2, 1, 1))).astype(int)
                u = 2 * np.random.rand(dim)
                w = Unifrnd(0, 2, 1, dim).reshape(-1, 1)

                A, B, C = RndX(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3 

                beta = np.random.rand(dim)
                Xnew1 = (beta * Best_X + (1 - beta) * Xavg).reshape(-1, 1)

                Xnew2 = np.zeros(dim)
                for j in range(dim):
                    if w[j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[j] * np.abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * np.abs((u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

                FU = Xnew2 > ub
                FL = Xnew2 < lb

                Xnew2 = (Xnew2 * (~(FU + FL))) + ub * FU + lb * FL
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else:
                    if np.random.rand() < w[np.random.randint(dim)]:
                        SM = RungeKutta(X[i, :], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM + (2 * np.random.rand(dim) * Best_X - Xnew2))

                        FU = Xnew > ub
                        FL = Xnew < lb
                        Xnew = ((Xnew * (~(FU + FL))) + ub * FU + lb * FL).reshape(-1, 1)
                        CostNew = fobj(Xnew)

                        if CostNew < Cost[i]:
                            shape = np.array(Xnew).shape
                            if shape[0] == 1:
                                X[i, :] = Xnew
                            elif shape[1] == 1:
                                X[i, :] = Xnew.reshape(1, -1)
                            Cost[i] = CostNew

                if Cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]

        Convergence_curve[0][it-1] = Best_Cost
        print(f"it:{it},Best Cost={Convergence_curve[0][it-1]}")

    return Best_Cost,Best_X,Convergence_curve[0]