# Windows大模型学习环境搭建完全指南

> **适用人群**: 零基础用户  
> **系统要求**: Windows 10/11  
> **显卡要求**: 4GB显存及以上(支持CPU运行)  
> **预计时间**: 2-3小时

---

## 📋 目录

- [环境准备](#环境准备)
- [核心组件安装](#核心组件安装)
- [开发工具配置](#开发工具配置)
- [运行你的第一个大模型](#运行你的第一个大模型)
- [常见问题](#常见问题)
- [进阶选项](#进阶选项)

---

## 环境准备

### 系统要求检查

**操作系统版本**

按 `Win + R`,输入 `winver` 查看系统版本:
- Windows 10 (1909+) ✅
- Windows 11 ✅

**硬件配置建议**

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 Intel i5/AMD R5 | 8核 Intel i7/AMD R7 |
| 内存 | 8GB | 16GB+ |
| GPU | GTX 1650 (4GB) | RTX 3060 (12GB) |
| 硬盘 | 300GB 可用空间 | SSD |

> **提示**: 没有NVIDIA显卡也可以继续学习,使用CPU运行(速度较慢但可用)

---

## 核心组件安装

### 1️⃣ NVIDIA显卡驱动(GPU用户)

**检查显卡型号**

在命令提示符中执行:
```cmd
wmic path win32_VideoController get name
```

或通过设备管理器: `右键此电脑` → `管理` → `设备管理器` → `显示适配器`

**下载驱动**

访问 [NVIDIA驱动下载页](https://www.nvidia.com/Download/index.aspx)

1. 选择显卡型号(如 GeForce GTX 1650)
2. 选择操作系统: Windows 10/11 64-bit
3. 下载并运行安装程序
4. 选择"自定义安装" → 勾选"执行干净安装"
5. **安装完成后重启电脑**

**验证安装**

重启后打开命令提示符:
```cmd
nvidia-smi
```

成功输出示例:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
+-----------------------------------------------------------------------------+
```

> **注意**: 记住显示的 `CUDA Version`,后续安装时需要

---

### 2️⃣ Miniconda环境管理器

**下载安装**

1. 访问 [Miniconda官网](https://docs.anaconda.com/miniconda/)
2. 下载 **Miniconda3 Windows 64-bit** (~80MB)
3. 运行安装程序
4. **重要**: 勾选 "Add Miniconda3 to my PATH environment variable"
5. 勾选 "Register Miniconda3 as my default Python"

**验证安装**

打开**新的**命令提示符窗口:
```cmd
conda --version
python --version
```

预期输出:
```
conda 24.x.x
Python 3.11.x
```

**配置国内镜像**(加速下载)

```cmd
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

---

### 3️⃣ CUDA Toolkit(GPU用户,可选)

**确定版本**

根据 `nvidia-smi` 显示的 CUDA Version:
- 显示 12.x → 安装 CUDA 11.8 或 12.1
- 显示 11.x → 安装 CUDA 11.8
- **推荐: CUDA 11.8**(兼容性最佳)

**下载安装**

1. 访问 [CUDA工具包归档](https://developer.nvidia.com/cuda-toolkit-archive)
2. 选择版本(如 CUDA Toolkit 11.8.0)
3. 选择: Windows → x86_64 → 10/11 → exe(local)
4. 下载 (~3GB) 并运行
5. 选择"自定义安装",保持默认组件
6. 等待安装完成(10-15分钟)

**验证安装**

```cmd
nvcc --version
```

预期输出:
```
Cuda compilation tools, release 11.8, V11.8.89
```

---

### 4️⃣ 创建Python虚拟环境

**创建环境**

```cmd
conda create -n llm python=3.10 -y
```

- `llm` 是环境名称(可自定义)
- `python=3.10` 指定Python版本

**激活环境**

```cmd
conda activate llm
```

成功后命令行前缀显示: `(llm) C:\Users\...>`

**常用命令**

```cmd
# 查看所有环境
conda env list

# 退出当前环境
conda deactivate

# 删除环境
conda remove -n llm --all
```

---

### 5️⃣ 安装PyTorch

**获取安装命令**

访问 [PyTorch官网](https://pytorch.org/get-started/locally/)

选择配置:
- PyTorch Build: **Stable**
- OS: **Windows**
- Package: **Conda**
- Language: **Python**
- Compute Platform: 
  - 有NVIDIA GPU → 选择对应CUDA版本
  - 仅CPU → 选择 **CPU**

**执行安装**

确保已激活 `llm` 环境:

```cmd
# GPU版本 (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或 CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

安装时间: 10-20分钟

**验证安装**

创建测试文件 `test_pytorch.py`:

```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU测试
    x = torch.rand(1000, 1000).cuda()
    print("✅ GPU加速正常工作")
else:
    print("ℹ️ 当前使用CPU模式")
```

运行测试:
```cmd
python test_pytorch.py
```

---

## 开发工具配置

### VSCode编辑器

**下载安装**

1. 访问 [VSCode官网](https://code.visualstudio.com/)
2. 下载并运行安装程序
3. **重要选项**:
   - ✅ 添加到PATH
   - ✅ 通过Code打开(右键菜单)
   - ✅ 注册为支持的文件类型编辑器

**安装扩展**

在VSCode中按 `Ctrl + Shift + X`,搜索并安装:
1. **Python** (Microsoft)
2. **Jupyter** (Microsoft)
3. **Pylance** (通常自动安装)
4. **Chinese Language Pack**(可选)

**选择Python解释器**

1. 按 `Ctrl + Shift + P`
2. 输入 "Python: Select Interpreter"
3. 选择 `llm` 环境

---

### Jupyter Notebook

**安装**

```cmd
conda activate llm
conda install jupyter -y
```

**启动方式**

方式一 - 命令行:
```cmd
jupyter notebook
```

方式二 - VSCode:
1. 创建 `.ipynb` 文件
2. 选择 `llm` 内核
3. 开始编写代码

---

## 运行你的第一个大模型

### 安装依赖库

```cmd
conda activate llm
pip install transformers accelerate sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 推荐模型(4GB显存)

| 模型 | 参数量 | 显存占用 | 下载大小 | 质量 |
|------|--------|----------|----------|------|
| TinyLlama-1.1B | 1.1B | ~2.5GB | ~2GB | 适合入门 |
| Qwen2.5-1.5B | 1.5B | ~3.5GB | ~3GB | 质量更好 |

### 实战: TinyLlama推理

创建 `demo_tinyllama.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型配置
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("加载模型中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"✅ 模型已加载到: {model.device}")

# 推理函数
def chat(prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试对话
questions = [
    "介绍一下Python编程语言",
    "什么是人工智能?",
    "写一个Hello World程序"
]

for i, q in enumerate(questions, 1):
    print(f"\n问题{i}: {q}")
    response = chat(q)
    print(f"回答: {response}\n" + "-"*50)

# 显存使用
if torch.cuda.is_available():
    print(f"\n显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

运行:
```cmd
python demo_tinyllama.py
```

### 实战: Qwen2.5多轮对话

创建 `demo_qwen.py`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("加载Qwen2.5模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ 加载完成\n")

def chat(messages):
    """多轮对话函数"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    return response

# 多轮对话演示
conversation = []

# 第一轮
user_input = "用Python写一个斐波那契数列函数"
conversation.append({"role": "user", "content": user_input})
print(f"用户: {user_input}")

assistant_reply = chat(conversation)
conversation.append({"role": "assistant", "content": assistant_reply})
print(f"Qwen: {assistant_reply}\n")

# 第二轮
user_input = "如何优化这个函数的性能?"
conversation.append({"role": "user", "content": user_input})
print(f"用户: {user_input}")

assistant_reply = chat(conversation)
print(f"Qwen: {assistant_reply}")
```

运行:
```cmd
python demo_qwen.py
```

### 配置Hugging Face镜像

如果下载速度慢,配置环境变量:

**临时设置**(当前终端):
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

**永久设置**:
1. 右键"此电脑" → 属性 → 高级系统设置
2. 环境变量 → 用户变量 → 新建
3. 变量名: `HF_ENDPOINT`
4. 变量值: `https://hf-mirror.com`

---

## 常见问题

### ❌ conda命令无法识别

**原因**: 环境变量未配置

**解决**:
1. 手动添加环境变量:
   - 右键"此电脑" → 属性 → 环境变量
   - 编辑用户变量 `Path`
   - 添加路径: `C:\Users\你的用户名\miniconda3`
   - 添加路径: `C:\Users\你的用户名\miniconda3\Scripts`
2. 重启命令提示符

### ❌ CUDA out of memory

**原因**: 显存不足

**解决方案**(按优先级):
1. 使用更小的模型(如TinyLlama)
2. 降低batch size为1
3. 清理显存缓存:
   ```python
   torch.cuda.empty_cache()
   ```
4. 切换到CPU模式:
   ```python
   model = model.to("cpu")
   ```

### ❌ 模型下载失败

**原因**: 网络连接问题

**解决**:
1. 使用镜像站(见上文配置)
2. 或手动下载:
   - 访问 https://hf-mirror.com
   - 搜索模型名称
   - 下载所有文件到本地
   - 使用本地路径加载:
     ```python
     model = AutoModelForCausalLM.from_pretrained("./local_path")
     ```

### ❌ VSCode找不到conda环境

**解决**:
1. 按 `Ctrl + Shift + P`
2. 输入 "Python: Select Interpreter"
3. 点击刷新图标
4. 选择 `llm` 环境

### ❌ pip安装太慢

**解决**: 配置国内镜像

```cmd
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 进阶选项

### 使用官方Python(替代Conda)

**安装步骤**

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载最新Python 3.x版本
3. 安装时勾选 **"Add Python to PATH"**
4. 验证: `python --version`

**创建虚拟环境**

```cmd
python -m venv llm-env
llm-env\Scripts\activate
```

### 使用uv工具(高级)

**安装uv**

PowerShell执行:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**创建并使用环境**

```cmd
# 创建环境
uv venv .venv

# 激活环境
.venv\Scripts\activate

# 安装PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
uv pip install transformers accelerate
```

**优势**: 安装速度更快,依赖管理更简洁

---

## ✅ 完整验证清单

按顺序执行以下命令,确保环境正常:

```cmd
# 1. 驱动检查(GPU用户)
nvidia-smi

# 2. Conda检查
conda --version

# 3. Python检查
python --version

# 4. CUDA检查(GPU用户)
nvcc --version

# 5. 激活环境
conda activate llm

# 6. PyTorch检查
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 7. Transformers检查
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

全部正常输出 → 🎉 环境搭建成功!

---

## 下一步

现在你可以:
- ✅ 运行各种开源大模型
- ✅ 学习模型微调技术
- ✅ 进行推理优化实验
- ✅ 参与开源项目

**推荐资源**:
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [Hugging Face课程](https://huggingface.co/course)
- [Transformers文档](https://huggingface.co/docs/transformers)

祝学习愉快! 🚀