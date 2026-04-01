import torch, time

device = "cuda"

n = 10000
a = torch.randn(n,n, device=device)
b = torch.randn(n,n, device=device)

torch.cuda.synchronize()
t0 = time.time()

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()
print("time:", time.time()-t0)
