import numpy as np
from src.linear.regressor import Regressor

class RidgeRegressor(Regressor):
    """
    Ridge regression model
    w* = argmin |t - X @ w| + a*|w|_2^2
    """
