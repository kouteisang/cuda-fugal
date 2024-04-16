import numpy as np
from scipy.special import logsumexp

n = 37
k = np.zeros((37, 37))

for i in range(37):
    for j in range(37):
        k[i, j] = i*n+j;

b = np.ones(37, dtype=np.float32)
u = np.ones(37,dtype=np.float32)/n

print(np.log(b) - logsumexp(k+np.log(u), axis = 1))
