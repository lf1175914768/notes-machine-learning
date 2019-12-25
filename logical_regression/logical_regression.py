import time
import numpy as np

def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc

def loadDataSet(filename, addones=False):
    """
    读取数据集
    :param addones: 是否添加 1, The default is not
    :param filename: 文件名
    :return:
        X: 训练样本集矩阵
        y: 标签集矩阵
    """
    file = open(filename)
    numFeat = len(file.readline().split('\t')) - 1
    X = []
    y = []
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        if addones:
            X.append([1.0, float(lineArr[0]), float(lineArr[1])])
        else:
            X.append([float(lineArr[0]), float(lineArr[1])])
        y.append(float(curLine[-1]))
    return np.mat(X), np.mat(y).T


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def J(theta, X, y, theLambda=0):
    """预测代价函数
    """
    m, n = X.shape
    h = sigmoid(X.dot(theta))
    J = (-1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (theLambda/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return(np.inf)
    return J.flatten()[0,0]


@exeTime
def gradient(X, y, options):
    """
    随机梯度下降法
    :param X: 样本矩阵
    :param y: 标签矩阵
    :param options: 选项
    :return: (thetas, errors), timeConsumed
    """
    m, n = X.shape
    # 初始化参数矩阵
    theta = np.ones((n, 1))
    count = 0    # 迭代次数
    # 初始化误差无限大
    error = float('inf')
    rate = options.get('rate', 0.01)
    epsilon = options.get('eosilon', 0.1)
    maxLoop = options.get('maxLoop', 1000)
    theLambda = options.get('theLambda', 0)
    method = options['method']
    errors = []   # 保存误差变化情况
    thetas = []  # 保存参数变化情况
    def _sgd(theta):
        converged = False
        for i in range(maxLoop):
            if converged:
                break
            for j in range(m):
                h = sigmoid(X[j] * theta)
                diff = h - y[j]
                theta = theta - rate * (1.0 / m) * X[j].T * diff
                error = J(theta, X, y)
                errors.append(error)
                if error < epsilon:
                    converged = True
                    break
                thetas.append(theta)
        return thetas, errors, i + 1
    def _bgd(theta):
        for i in range(maxLoop):
            h = sigmoid(X.dot(theta))
            diff = h - y
            # theta0 should not be regularized
            theta = theta - rate * ((1.0 / m)*X.T*diff + (theLambda/m)*np.r_[[[0]], theta[1:]])
            error = J(theta, X, y, theLambda)
            errors.append(error)
            if error < epsilon:
                break
            thetas.append(theta)
        return thetas, errors, i + 1
    methods = {
        'sgd': _sgd,
        'bgd': _bgd
    }
    return methods[method](theta)


def oneVsAll(X, y, options):
    """
    多分类
    :param X: 样本
    :param y: 标签
    :param options: 选项
    :return: 权值矩阵
    """
    # 类型数
    classes = set(np.ravel(y))
    # 决策边界矩阵
    Thetas = np.zeros((len(classes), X.shape[1]))
    # 一次选定每种分类对应的样本为正样本，其他样本标识为负样本，进行逻辑回归
    for idx, c in enumerate(classes):
        newY = np.zeros(y.shape)
        newY[np.where(y == c)] = 1
        result, timeConsumed = gradient(X, newY, options)
        thetas, errors, iterations = result
        Thetas[idx] = thetas[-1].ravel()
    return Thetas


def predictOneVsAll(X, thetas):
    H = sigmoid(thetas * X.T)
    return H