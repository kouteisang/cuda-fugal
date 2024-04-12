import torch


# device = torch.device("cuda")

device = torch.device("cuda")

na = 4
nb = 4

a = torch.ones(na, dtype = torch.float32).to(device)
b = torch.ones(na, dtype = torch.float32).to(device)

u = torch.ones(na, dtype=torch.float32).to(device) / na
v = torch.ones(nb, dtype=torch.float32).to(device) / nb

K = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=torch.float32).to(device)

for i in range(100):

    KTu = torch.matmul(u, K)

    v = torch.div(b, KTu)

    Kv = torch.matmul(K, v)

    u = torch.div(a, Kv)

res = u.reshape(-1, 1) * K * v.reshape(1, -1) 
print(torch.sum(res, dim = 0))
print(torch.sum(res, dim = 1))
