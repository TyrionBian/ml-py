

import numpy as np

a = np.random.rand(8,13)
b = np.random.rand(13,8)
c = a @ b  # Python 3.5+
d = np.dot(a, b)

print(c.shape, d.shape)