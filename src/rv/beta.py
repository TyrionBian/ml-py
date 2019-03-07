import numpy as np
from scipy.special import gamma
from prml.rv.rv import RandomVariable


np.seterr(all="ignore")

class Beta(RandomVariable):
    """
    Beta distribution
    p(mu|n_ones, n_zeros)
    = gamma(n_ones + n_zeros)
      * mu^(n_ones - 1) * (1 - mu)^(n_zeros - 1)
      / gamma(n_ones) / gamma(n_zeros)
    """
    