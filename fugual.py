import numpy as np
import torch
from torch.utils.cpp_extension import load, load_inline


cuda_kernel = load(name="cuda_kernel",
                  extra_include_paths=["./"] ,
                  sources=["sinkhorn_log.cu", "glue.cpp"],
                  verbose=True)

device = torch.device('cuda:0')

N = 3
# cost = torch.rand((N, N), dtype=torch.float, device=device)
cost = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float, device=device)
cost_t = torch.zeros((N, N), dtype=torch.float, device=device)
# u = torch.ones(N).to(device)/N
# v = torch.ones(N).to(device)/N

# print(u)
# cuda_kernel.sinkhorn_log_kernel(cost, v, u, N)
# print(u)

print(cost)
print(cost_t)


cuda_kernel.matrix_transpose_kernel(cost_t, cost, N)

print(cost_t)