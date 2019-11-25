import numpy as np

def euc2(x, y):
    # calculate squared Euclidean distance

    # check dimension
    assert x.shape == y.shape  

    diff = x - y

    return np.dot(diff, diff)


def mah2(x, y, Sigma):
    # calculate squared Mahalanobis distance

    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    
    diff = x - y
    
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)

