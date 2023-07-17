import pytest
import torch
import time

def test_is_cuda_available():
    assert torch.cuda.is_available()


def test_cuda_speed():
    print(f"Preparing the tensors")
    size = 4000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU
    print(f"Running on CPU")
    start = time.time()
    torch.matmul(a, b)
    end = time.time()
    cpu_time = end-start
    print(f"CPU time: {cpu_time}s")

    # GPU
    a, b = a.cuda(), b.cuda()
    print(f"Running on gpu:", torch.cuda.get_device_name(0))
    start = time.time()
    torch.matmul(a, b)
    end = time.time()
    gpu_time = end-start
    print(f"GPU time: {gpu_time}s")

    assert gpu_time < cpu_time/2
