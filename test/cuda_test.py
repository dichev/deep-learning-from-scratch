import torch
import time

print(f"Preparing the tensors")
size = 10000
a = torch.randn(size, size)
b = torch.randn(size, size)

# CPU
print(f"Running on CPU")
start = time.time()
torch.matmul(a, b)
end = time.time()
print(f"CPU time: {end - start}s")

# GPU
if torch.cuda.is_available():
    print(f"Running on gpu:", torch.cuda.get_device_name(0))
    a = a.cuda()
    b = b.cuda()
    start = time.time()
    torch.matmul(a, b)
    end = time.time()
    print(f"GPU time: {end - start}s")



