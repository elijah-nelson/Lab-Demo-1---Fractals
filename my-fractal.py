"""
A visualisation of the Feigenbaum attractor
 - See Wikipedia page https://en.wikipedia.org/wiki/Logistic_map
@author Elijah Nelson
"""

import torch
import numpy as np

print("PyTorch Version:", torch.__version__)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

num_ks = 1000 # number of ks to test
num_x0s = 100
n = 500      # number of iterations

K = np.linspace(0.0, 4.0, num_ks)   # [k0, k1, ..., kn]
K = K.repeat(num_x0s)               # [k0, k0, k0, ..., k0 {num_x0s times}, k1, ..., k1, ..., kn]
X = np.linspace(0.01, 0.99, num_x0s)# [x00, x01, x02, x03, ..., x0n]
X = np.tile(X, num_ks)              # [x00, x01, ..., x0n, x00, x01, x0n, ...] repeated num_ks times

X = torch.Tensor(X)
K = torch.Tensor(K)

# transfer to the GPU device
X = X.to(device)
K = K.to(device)

for _ in range(n):
    X = K * X * (1 - X) # parallel!

import matplotlib.pyplot as plt

plt.scatter(K.numpy(), X.numpy(), s=plt.rcParams['lines.markersize'] / 16, c=K.numpy())
plt.xlabel('k')
plt.ylabel(f'x after {n} iterations')
plt.show()