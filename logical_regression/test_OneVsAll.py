import numpy as np
import logical_regression as regression
from scipy.io import loadmat

if __name__ == '__main__':
    data = loadmat('data/ex3data1.mat')
    X = np.mat(data['X'])
    y = np.mat(data['y'])
    # 为X添加偏置
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    options = {
        'rate':0.1,
        'epsilon':0.1,
        'maxLoop':4000,
        'method':'bgd'
    }
    # training
    thetas = regression.oneVsAll(X, y, options)
    # predict
    H = regression.predictOneVsAll(X, thetas)
    pred = np.argmax(H, axis=0) + 1
    # calculate accuracy
    print('Training accuracy is : %.2f%' % (np.mean(pred == y.ravel()) * 100))