import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

"""
Svm 模型
参考 https://zhuanlan.zhihu.com/p/29212107 实现的过程
"""


def linearKernel():
    """
    线性核函数
    :return:
    """

    def calc(X, A):
        return X * A.T

    return calc


def rbfKernel(delta):
    """
    rbf 核函数
    :return:
    """
    gamma = 1.0 / (2 * delta ** 2)

    def calc(X, A):
        return np.mat(rbf_kernel(X, A, gamma=gamma))

    return calc


def getSmo(X, y, C, tol, maxIter, kernel=linearKernel()):
    """
    SMo
    :param X: 训练样本
    :param y: 标签集
    :param C: 正规化参数
    :param tol: 容忍值
    :param maxIter: 最大迭代次数
    :param kernel: 所用核函数
    :return:
    """
    m, n = X.shape
    K = kernel(X, X)
    # cache 用于存放预测误差，用以加快计算速度
    Ecache = np.zeros((m, 2))

    def predict(X, alphas, b, supportVectorsIndex, supportVectors):
        """
        计算权值向量
        :param X: 预测矩阵
        :param alphas:
        :param b:
        :param supportVectorsIndex: 支持向量坐标集
        :param supportVectors: 支持向量
        :return: 预测结果
        """
        Ks = kernel(supportVectors, X)
        predicts = (np.multiply(alphas[supportVectorsIndex], y[supportVectorsIndex]).T * Ks + b).T
        predicts = np.sign(predicts)
        return predicts

    def w(alphas, b, supportVectorsIndex, supportVectors):
        """计算权值
        """
        return (np.multiply(alphas[supportVectorsIndex], y[supportVectorsIndex]).T * supportVectors).T

    def E(i, alphas, b):
        """计算预测误差
        """
        FXi = float(np.multiply(alphas, y).T * K[:, i]) + b
        E = FXi - float(y[i])
        return E

    def updateE(i, alphas, b):
        Ecache[i] = [1, E(i, alphas, b)]

    def selectJRand(i):
        """
        """
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def selectJ(i, Ei, alphas, b):
        # 选择权值
        maxJ = 0;
        maxDist = 0;
        Ej = 0
        Ecache[i] = [1, Ei]
        validCaches = np.nonzero(Ecache[:, 0])[0]
        if len(validCaches) > 1:
            for k in validCaches:
                if k == i:
                    continue
                Ek = E(k, alphas, b)
                dist = np.abs(abs(Ei - Ek))
                if maxDist < dist:
                    Ej = Ek
                    maxJ = k
                    maxDist = dist
            return maxJ, Ej
        else:
            ### 随机选择
            j = selectJRand(i)
            Ej = E(j, alphas, b)
            return j, Ej

    def select(i, alphas, b):
        # alpha 对选择
        Ei = E(i, alphas, b)
        # 选择违背KKT条件的， 作为alpha2
        Ri = y[i] * Ei
        # 关于这里的解释：
        """
        KKT条件： 
        alphas[i] = 0        <=>    y[i] * g(x[i]) >= 1
        0 < alphas[i] < C    <=>    y[i] * g(x[i]) = 1
        alphas[i] = C        <=>    y[i] * g(x[i])  <= 1
        也 即 违反KKT 条件的公式等价于:
        1、 y[i] * g(x[i]) < 1时， 0 <= alphas[i] < C 
        2、 y[i] * g(x[i]) > 1时， 0 < alphas[i] <= C
        ##### NOTICE: y[i] * g(x[i]) = 1 时， 0 <= alphas[i]  <= C 总是成立的
        
        这里 Ei = g(x[i]) - y[i],   y[i] * y[i] = 1
        (Ri = y[i] * (g(x[i]) - y[i])) < -tol and alphas[i] < C    <=>   y[i] * g(x[i]) < 1 - tol时， 0 <= alphas[i] < C
        (Ri = y[i] * (g(x[i]) - y[i])) > tol and alphas[i] > 0    <=>   y[i] * g(x[i]) > 1 + tol时， 0 < alphas[i] <= C
        综上，多出的一个 tol 为误差项
        """
        if (Ri < -tol and alphas[i] < C) or \
                (Ri > tol and alphas[i] > 0):
            # 选择第二个参数
            j = selectJRand(i)
            Ej = E(j, alphas, b)
            # j, Ej = selectJ(i, Ei, alphas, b)
            # get bounds
            if y[i] != y[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                return 0, alphas, b
            Kii = K[i, i]
            Kjj = K[j, j]
            Kij = K[i, j]
            eta = 2.0 * Kij - Kii - Kjj
            if eta >= 0:
                return 0, alphas, b
            iOld = alphas[i].copy()
            jOld = alphas[j].copy()
            alphas[j] = jOld - y[j] * (Ei - Ej) / eta
            if alphas[j] > H:
                alphas[j] = H
            elif alphas[j] < L:
                alphas[j] = L
            if abs(alphas[j] - jOld) < tol:
                alphas[j] = jOld
                return 0, alphas, b
            alphas[i] = iOld + y[i] * y[j] * (jOld - alphas[j])
            # update Ecache
            updateE(i, alphas, b)
            updateE(j, alphas, b)
            # update b
            bINew = b - Ei - y[i] * (alphas[i] - iOld) * Kii - y[j] * \
                    (alphas[j] - jOld) * Kij
            bJNew = b - Ej - y[i] * (alphas[i] - iOld) * Kij - y[j] * \
                    (alphas[j] - jOld) * Kjj
            if alphas[i] > 0 and alphas[i] < C:
                bNew = bINew
            elif alphas[j] > 0 and alphas[j] < C:
                bNew = bJNew
            else:
                bNew = (bINew + bJNew) / 2
            return 1, alphas, b
        else:
            return 0, alphas, b

    def train():
        """
        完整版训练算法
        :return:
        """
        numChanged = 0
        examineAll = True
        iterCount = 0
        b = 0
        alphas = np.mat(np.zeros((m, 1)))
        while (numChanged > 0 or examineAll) and (iterCount < maxIter):
            numChanged = 0
            if examineAll:
                for i in range(m):
                    changed, alphas, b = select(i, alphas, b)
                    numChanged += changed
            else:
                nonBoundIds = np.nonzero((alphas.A > 0) * (alphas.A < C))[0]
                for i in nonBoundIds:
                    changed, alphas, b = select(i, alphas, b)
                    numChanged += changed
            iterCount += 1

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
        supportVectorsIndex = np.nonzero(alphas.A > 0)[0]
        supportVectors = np.mat(X[supportVectorsIndex])
        return alphas, w(alphas, b, supportVectorsIndex, supportVectors), b, \
                supportVectorsIndex, supportVectors, iterCount


    def trainSimple():
        """
        简化版训练算法
        :return:
        """
        numChanged = 0
        alphas = np.mat(np.zeros((m, 1)))
        b = 0; L = 0; H = 0
        iterCount = 0
        while iterCount < maxIter:
            numChanged = 0
            for i in range(m):
                Ei = E(i, alphas, b)
                Ri = y[i] * Ei
                # 选择违背KKT条件的，作为alpha2
                if (Ri < -tol and alphas[i] < C) or \
                        (Ri > tol and alphas[i] > 0):
                    # 选择第二个参数
                    j = selectJRand(i)
                    Ej = E(j, alphas, b)
                    # get bounds
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    Kii = K[i, i]
                    Kjj = K[j, j]
                    Kij = K[i, j]
                    eta = 2.0 * Kij - Kii - Kjj
                    if eta >= 0:
                        continue
                    iOld = alphas[i].copy()
                    jOld = alphas[j].copy()
                    alphas[j] = jOld - y[j] * (Ei - Ej) / eta
                    if alphas[j] > H:
                        alphas[j] = H
                    elif alphas[j] < L:
                        alphas[j] = L
                    if abs(alphas[j] - jOld) < tol:
                        alphas[j] = jOld
                        continue
                    alphas[i] = iOld + y[i] * y[j] * (jOld - alphas[j])
                    # update b
                    bINew = b - Ei - y[i] * (alphas[i] - iOld) * Kii - y[j] * (alphas[j] - jOld) * Kij
                    bJNew = b - Ej - y[i] * (alphas[i] - iOld) * Kij - y[j] * (alphas[j] - jOld) * Kjj
                    if alphas[i] > 0 and alphas[i] < C:
                        b = bINew
                    elif alphas[j] > 0 and alphas[j] < C:
                        b = bJNew
                    else:
                        b = (bINew + bJNew) / 2.0
                    numChanged += 1
            if numChanged == 0:
                iterCount += 1
            else:
                iterCount = 0
        supportVectorsIndex = np.nonzero(alphas.A > 0)[0]
        supportVectors = np.mat(X[supportVectorsIndex])
        return alphas, w(alphas, b, supportVectorsIndex, supportVectors), b, \
               supportVectorsIndex, supportVectors, iterCount

    return trainSimple, train, predict
