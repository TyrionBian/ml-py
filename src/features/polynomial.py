import itertools
import functools
import numpy as np

class PolynomialFeatures(object):
    """
    polynomial features :
    transforms input arrays with polynomial features
    """
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim==1:
            x = x[:,None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree+1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x,y: x*y, items ))
        return np.asarray(features).transpose()
        