import torch

print("PyTorch基础练习")
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
c = torch.mm(a, b)
print(f"矩阵乘法结果:\n{c}")

x = torch.randn(1000, 1000)
y = torch.relu(x)
print(f"ReLU结果形状: {y.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.randn(3, 3).to(device)
print(f"张量设备: {t.device}")

print("完成")