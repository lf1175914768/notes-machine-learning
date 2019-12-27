import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import smo


data = loadmat('data/ex6data1.mat')

X = np.mat(data['X'])
y = np.mat(data['y'], dtype=np.float)
y[y == 0] = -1

m, n = X.shape
tol = 1e-3
maxIter = 20
C = 100.0

trainSimple, train, predict = smo.getSmo(X, y, C, tol, maxIter)
alphas, w, b, supportVectorsIndex, supportVectors, iterCount = trainSimple()
print(w)
print(b)
print(len(supportVectorsIndex))
print('iterCount: %d' % iterCount)

# predictions = predict(X, alphas, b, supportVectorsIndex, supportVectors)

# 绘制数据点
x1Min = X[:, 0].min()
x1Max = X[:, 0].max()
x2Min = X[:, 1].min()
x2Max = X[:, 1].max()

plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x1Min - 1, x1Max + 1)
plt.ylim(x2Min - 1, x2Max + 1)
plt.title('C = %.1f' % C)
for i in range(m):
    x = X[i].A[0]
    if y[i] == 1:
        color = 'black'
        if i in supportVectorsIndex:
            color = 'red'
        plt.scatter(x[0], x[1], marker='*', color=color, s=50)
    else:
        color = 'green'
        if i in supportVectorsIndex:
            color = 'red'
        plt.scatter(x[0], x[1], marker='o', color=color, s=50)

plt.show()