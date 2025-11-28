import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
else:
    print("当前使用CPU模式")

x = torch.rand(3, 3)
print(f"\n随机张量:\n{x}")
if torch.cuda.is_available():
    x = x.cuda()
    print(f"张量已移至GPU: {x.device}")