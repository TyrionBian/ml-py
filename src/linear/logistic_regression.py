import numpy as np
from src.linear.classifier import Classifier

class LogisticRegression(Classifier):
    """
    Logistic regression model
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    @staticmethod
    def _sigmoid(x):
        # (tanh+1)/2
        return(1/np.exp(-x))

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        maximum likelihood estimation of logistic regression model
        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        w = np.zeros(np.size(X, axis=1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t)
            hession = X.T @ (y * (1 - y)) @ X
            try:
                w -= np.linalg.solve(hession, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def prob(self, X:np.ndarray):
        return self._sigmoid(X @ self.w)
    
    def classify(self, X:np.ndarray, threshold:float=0.5):
        return(self.prob(X) > threshold).astype(int)

