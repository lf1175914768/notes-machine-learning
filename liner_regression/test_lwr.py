import regression
import numpy as np

if __name__ == "__main__":
    srcX, y = regression.loadDataSet("data/lwr.txt")
    m, n = srcX.shape
    srcX = np.concatenate((srcX[:, 0], np.power(srcX[:, 0], 2)), axis=1)
    # 特征缩放
    X = regression.standarize(srcX.copy())
