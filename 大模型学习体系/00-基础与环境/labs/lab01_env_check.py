import platform
import torch

print("环境检查开始")
print(f"Python版本: {platform.python_version()}")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")

try:
    import transformers
    print(f"Transformers版本: {transformers.__version__}")
except Exception:
    print("Transformers未安装或导入失败")

x = torch.rand(2, 2)
print(f"随机张量:\n{x}")
if torch.cuda.is_available():
    x = x.cuda()
    print(f"张量设备: {x.device}")

print("环境检查完成")