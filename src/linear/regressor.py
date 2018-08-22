import numpy as np

class Regressor (object):
    """Base class for regression"""
    def fit(self, X, t, **kwargs):
        """
        estimate parameters given train datasets
        X : input dataset, np.ndarray
        t : target, np.ndarray
        """
        self._check_input(X)
        self._check_target(t)
        if hasattr(self, "_fit"):
            self._fit(X, t, **kwargs)
        else :
            raise NotImplementedError
    
    def predict(self, X, **kwargs):
        """
        predict outputs of the model
        """
        self._check_input(X)
        if hasattr(self, "_predict"):
            return self._predict(X, **kwargs)
        else :
            raise NotImplementedError

    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("input dataset X is not np.ndarray. ")
        if X.ndim != 2:
            raise ValueError("the dim of X is not 2")
        if hasattr(self, "n_features") and self.n_features != np.size(X,axis=1):
            raise ValueError(
                "mismatch dimension of 1 of X(input) "
                "(size {} is different from {})"
                .format(np.size(X, 1), self.n_features)
            )
    
    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target  must be np.ndarray)")
        if t.ndim !=1:
            raise ValueError("the dim of t is not 1")
