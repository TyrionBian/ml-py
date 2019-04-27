import numpy as np

class PCA(object):

    def __init__(self, n_components):
        """
        construct principal component analysis
        Parameters
        ----------
        n_components : int
            number of components
        """
        assert isinstance(n_components, int)
        self.n_components = n_components

    def fit(self, X, method="eigen", iter_max=100):
        """
        maximum likelihood estimate of pca parameters
        x ~ \int_z N(x|Wz+mu,sigma^2)N(z|0,I)dz
        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        method : str
            method to estimate the parameters
            ["eigen", "em"]
        iter_max : int
            maximum number of iterations for em algorithm
        Attributes
        ----------
        mean : (n_features,) ndarray
            sample mean of the data
        W : (n_features, n_components) ndarray
            projection matrix
        var : float
            variance of observation noise
        C : (n_features, n_features) ndarray
            variance of the marginal dist N(x|mean,C)
        Cinv : (n_features, n_features) ndarray
            precision of the marginal dist N(x|mean, C)
        """
        