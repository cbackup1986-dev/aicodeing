# 第二章：模型部署（Windows 平台）

> **目标**：在 Windows 环境下完整演示 Docker Desktop 的安装配置流程，并详细说明如何使用 Docker 部署 vLLM、Xinference 以及 Ollama 等主流推理框架。

## 📋 目录

- [1. 先决条件](#1-先决条件)
- [2. WSL2 安装配置](#2-wsl2-安装配置)
- [3. Docker Desktop 安装](#3-docker-desktop-安装)
- [4. GPU 支持配置](#4-gpu-支持配置)
- [5. 模型文件管理](#5-模型文件管理)
- [6. vLLM 部署](#6-vllm-部署)
- [7. Xinference 部署](#7-xinference-部署)
- [8. Ollama 部署](#8-ollama-部署)
- [9. 验证与测试](#9-验证与测试)
- [10. 常见问题排查](#10-常见问题排查)

---

## 1. 先决条件

### 1.1 系统要求

**操作系统**
- Windows 11（推荐）或 Windows 10 21H2 及以上版本
- 已安装最新的 Windows 更新
- BIOS/UEFI 中已启用虚拟化支持（Intel VT-x 或 AMD-V）

**硬件配置**
- **CPU**: 现代多核处理器（建议 8 核及以上）
- **内存**: 至少 16GB RAM（推荐 32GB 用于大模型推理）
- **GPU**: NVIDIA 显卡（支持 CUDA，推荐 RTX 系列或专业卡）
- **磁盘**: 至少 100GB 可用空间（用于模型存储）

### 1.2 网络与凭证

**网络要求**
- 稳定的互联网连接（用于下载 Docker、模型文件）
- 能够访问 Hugging Face、Docker Hub 等资源

**可选凭证**
- Hugging Face Token（下载私有或门控模型时需要）
- Docker Hub 账号（避免镜像拉取限制）
- 私有镜像仓库凭证（如使用企业内部镜像）

### 1.3 架构选择说明

本指南采用 **WSL2 + Docker Desktop** 方案，原因如下：

✅ Linux 环境下容器运行更稳定  
✅ GPU 支持更成熟完善  
✅ 与生产环境（Linux）保持一致  
✅ 生态工具链更丰富

---

## 2. WSL2 安装配置

> 如果已安装 WSL2，可跳过此节直接进入 Docker 安装。

### 2.1 快速安装

以**管理员权限**打开 PowerShell，执行：

```powershell
wsl --install
```

该命令会自动完成以下操作：
- 启用 WSL 和虚拟机平台功能
- 下载并安装 Linux 内核更新
- 安装默认的 Ubuntu 发行版
- 设置 WSL 默认版本为 2

### 2.2 指定发行版安装

如需安装特定版本的 Ubuntu：

```powershell
# 查看可用发行版
wsl --list --online

# 安装指定版本
wsl --install -d Ubuntu-22.04
```

### 2.3 升级现有 WSL1 到 WSL2

如果已安装 WSL1，升级到 WSL2：

```powershell
# 设置默认版本为 WSL2
wsl --set-default-version 2

# 查看当前发行版版本
wsl --list --verbose

# 转换特定发行版到 WSL2
wsl --set-version Ubuntu 2
```

### 2.4 初始化配置

1. 从开始菜单打开 Ubuntu
2. 创建用户名和密码
3. 更新系统软件包：

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.5 验证安装

```powershell
# 在 PowerShell 中执行
wsl --status
wsl -l -v
```

期望输出示例：
```
  NAME            STATE           VERSION
* Ubuntu          Running         2
```

---

## 3. Docker Desktop 安装

### 3.1 下载安装包

访问 [Docker 官方网站](https://www.docker.com/products/docker-desktop/) 下载 Windows 版安装程序。

- 文件名：`Docker Desktop Installer.exe`
- 大小：约 500MB+
- 企业用户可使用内部镜像站

### 3.2 安装步骤

1. **以管理员身份运行**安装程序
2. 配置选项界面，确保勾选：
   - ✅ **Use WSL 2 instead of Hyper-V** (使用 WSL2 引擎)
   - ✅ **Add shortcut to desktop** (可选)
3. 安装程序会自动启用必要的 Windows 功能
4. 安装完成后，根据提示**重启计算机**

### 3.3 首次启动配置

启动 Docker Desktop 后进入设置：

#### 3.3.1 WSL 集成配置

1. 打开 **Settings** → **Resources** → **WSL Integration**
2. 启用集成：
   - ✅ **Enable integration with my default WSL distro**
   - ✅ **Ubuntu** (或你安装的其他发行版)
3. 点击 **Apply & Restart**

#### 3.3.2 资源限制（可选）

在 **Settings** → **Resources** → **Advanced** 中调整：

```yaml
CPUs: 8              # 分配给 Docker 的 CPU 核心数
Memory: 16 GB        # 分配的内存大小
Swap: 4 GB          # 交换空间
Disk image size: 100 GB  # 虚拟磁盘大小
```

### 3.4 验证安装

在 PowerShell 或 WSL 终端中执行：

```bash
# 检查版本
docker --version
docker compose version

# 运行测试容器
docker run --rm hello-world

# 查看系统信息
docker info
```

成功输出示例：
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

---

## 4. GPU 支持配置

> 仅适用于 NVIDIA GPU，AMD GPU 支持有限。

### 4.1 安装 NVIDIA 驱动

1. 访问 [NVIDIA 官方驱动页面](https://www.nvidia.com/Download/index.aspx)
2. 下载适用于你的显卡型号的**最新驱动**
3. 安装时选择"自定义安装"并勾选：
   - NVIDIA 图形驱动程序
   - PhysX 系统软件
   - CUDA 组件（如果可选）

**重要**：确保安装的是支持 WSL2 的驱动版本（通常 Release 470.xx 及以上）。

### 4.2 验证 GPU 在 WSL2 中可见

在 WSL2 终端（Ubuntu）中执行：

```bash
nvidia-smi
```

期望输出：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
```

### 4.3 验证 Docker GPU 支持

```bash
# 运行 NVIDIA CUDA 测试容器
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

如果能在容器内看到 GPU 信息，说明配置成功。

### 4.4 故障排查

**问题：WSL2 中 `nvidia-smi` 命令不存在**

解决方案：
```bash
# 在 WSL2 中不需要单独安装 CUDA toolkit
# 只需确保 Windows 上安装了支持 WSL 的 NVIDIA 驱动
```

**问题：Docker 容器看不到 GPU**

检查清单：
1. Windows 驱动版本是否支持 WSL2
2. Docker Desktop 是否使用 WSL2 后端
3. 是否使用了 `--gpus all` 参数

---

## 5. 模型文件管理

### 5.1 存储位置建议

**方案一：WSL2 文件系统内（推荐）**

```bash
# 在 WSL2 中创建模型目录
mkdir -p ~/models
```

优点：性能最佳，避免跨文件系统访问  
缺点：Windows 资源管理器访问不便

**方案二：Windows 文件系统**

```bash
# Windows 路径：C:\Models
# WSL2 访问路径：/mnt/c/Models
mkdir -p /mnt/c/Models
```

优点：Windows 下管理方便  
缺点：性能略低，可能有权限问题

### 5.2 从 Hugging Face 下载模型

#### 5.2.1 安装 Hugging Face CLI

```bash
# 在 WSL2 中执行
pip install -U "huggingface_hub[cli]"
```

#### 5.2.2 登录认证

```bash
huggingface-cli login
# 输入你的 Hugging Face Token
```

获取 Token：访问 https://huggingface.co/settings/tokens

#### 5.2.3 下载模型

```bash
# 下载完整模型仓库
huggingface-cli download \
  <repo-id> \
  --local-dir ~/models/<model-name> \
  --local-dir-use-symlinks False

# 示例：下载 Qwen2.5-7B-Instruct
huggingface-cli download \
  Qwen/Qwen2.5-7B-Instruct \
  --local-dir ~/models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

#### 5.2.4 使用 Git LFS（替代方案）

```bash
# 安装 Git LFS
sudo apt install git-lfs
git lfs install

# 克隆模型仓库
cd ~/models
git clone https://huggingface.co/<repo-id>
```

### 5.3 路径映射说明

在 Docker 中挂载模型目录：

```bash
# WSL2 路径挂载
docker run -v ~/models:/models <image>

# Windows 路径挂载
docker run -v /mnt/c/Models:/models <image>
```

---

## 6. vLLM 部署

### 6.1 镜像选择

vLLM 官方镜像：`vllm/vllm-openai:latest`

查看可用标签：https://hub.docker.com/r/vllm/vllm-openai/tags

### 6.2 使用 docker run 部署

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  -v ~/models:/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-7b \
  --trust-remote-code
```

**参数说明**：
- `--gpus all`: 分配所有 GPU
- `-v ~/models:/models`: 挂载模型目录
- `-p 8000:8000`: 端口映射
- `--ipc=host`: 共享主机 IPC 命名空间（提高性能）
- `--trust-remote-code`: 信任模型的远程代码

### 6.3 使用 Docker Compose 部署

创建 `docker-compose.yml`：

```yaml
version: "3.8"

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    restart: unless-stopped
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    
    volumes:
      - ~/models:/models
    
    ports:
      - "8000:8000"
    
    command:
      - --model
      - /models/Qwen2.5-7B-Instruct
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --served-model-name
      - qwen2.5-7b
      - --trust-remote-code
      - --max-model-len
      - "8192"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    ipc: host
```

启动服务：

```bash
docker compose up -d
```

### 6.4 验证部署

#### 6.4.1 检查容器状态

```bash
docker ps
docker logs vllm-server
```

#### 6.4.2 测试 API

```bash
# 查看可用模型
curl http://localhost:8000/v1/models

# 生成补全
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "你好，请介绍一下你自己。",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat 接口
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role": "user", "content": "什么是机器学习？"}
    ],
    "max_tokens": 200
  }'
```

### 6.5 性能优化参数

```bash
--model /models/Qwen2.5-7B-Instruct \
--tensor-parallel-size 2 \           # GPU 数量（张量并行）
--max-model-len 8192 \               # 最大序列长度
--gpu-memory-utilization 0.9 \       # GPU 内存利用率
--max-num-seqs 256 \                 # 最大并发序列数
--disable-log-requests \             # 禁用请求日志（提升性能）
--trust-remote-code
```

---

## 7. Xinference 部署

### 7.1 镜像选择

Xinference 官方镜像：`xprobe/xinference:latest`

### 7.2 使用 docker run 部署

```bash
docker run -d \
  --name xinference \
  --gpus all \
  -v ~/models:/root/.xinference/cache \
  -p 9997:9997 \
  --ipc=host \
  xprobe/xinference:latest \
  xinference-local --host 0.0.0.0 --port 9997
```

### 7.3 使用 Docker Compose 部署

创建 `docker-compose.yml`：

```yaml
version: "3.8"

services:
  xinference:
    image: xprobe/xinference:latest
    container_name: xinference-server
    restart: unless-stopped
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - XINFERENCE_HOME=/root/.xinference
    
    volumes:
      - ~/models:/root/.xinference/cache
      - xinference-data:/root/.xinference
    
    ports:
      - "9997:9997"
    
    command: xinference-local --host 0.0.0.0 --port 9997
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    ipc: host

volumes:
  xinference-data:
```

启动服务：

```bash
docker compose up -d
```

### 7.4 使用 Xinference

#### 7.4.1 访问 Web UI

浏览器访问：`http://localhost:9997`

#### 7.4.2 CLI 命令

```bash
# 列出可用模型
docker exec xinference xinference list --all

# 启动模型
docker exec xinference xinference launch \
  --model-name qwen2.5-instruct \
  --size-in-billions 7 \
  --model-format pytorch

# 查看运行中的模型
docker exec xinference xinference list
```

#### 7.4.3 API 调用

```bash
# 获取模型 UID（从 Web UI 或 CLI 获取）
MODEL_UID="<your-model-uid>"

# Chat 请求
curl -X POST http://localhost:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_UID'",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### 7.5 预加载模型（可选）

创建启动脚本 `start-with-model.sh`：

```bash
#!/bin/bash
xinference-local --host 0.0.0.0 --port 9997 &
sleep 10
xinference launch --model-name qwen2.5-instruct --size-in-billions 7 --model-format pytorch
wait
```

修改 Docker Compose：

```yaml
command: /bin/bash /app/start-with-model.sh
volumes:
  - ./start-with-model.sh:/app/start-with-model.sh
```

---

## 8. Ollama 部署

### 8.1 部署方案选择

**方案一：Docker 部署（推荐）**  
跨平台一致性好，易于管理

**方案二：Windows 原生安装**  
Ollama 已提供 Windows 安装包（较新版本）

**方案三：WSL2 原生安装**  
Linux 环境，性能最优

### 8.2 使用 Docker 部署 Ollama

```bash
docker run -d \
  --name ollama \
  --gpus all \
  -v ollama-data:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:latest
```

#### 8.2.1 拉取并运行模型

```bash
# 拉取模型
docker exec ollama ollama pull qwen2.5:7b

# 交互式运行
docker exec -it ollama ollama run qwen2.5:7b
```

#### 8.2.2 Docker Compose 配置

```yaml
version: "3.8"

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    restart: unless-stopped
    
    volumes:
      - ollama-data:/root/.ollama
    
    ports:
      - "11434:11434"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama-data:
```

### 8.3 API 调用示例

```bash
# 生成请求
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "为什么天空是蓝色的？",
  "stream": false
}'

# Chat 请求
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}'
```

### 8.4 Windows 原生安装（备选方案）

1. 访问 https://ollama.com/download
2. 下载 Windows 安装程序
3. 运行安装后，命令行执行：

```powershell
ollama --version
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

---

## 9. 验证与测试

### 9.1 容器健康检查

```bash
# 查看所有容器状态
docker ps -a

# 查看容器日志
docker logs <container-name>

# 实时查看日志
docker logs -f <container-name>

# 查看容器资源使用
docker stats
```

### 9.2 GPU 使用监控

```bash
# 在宿主机（WSL2）中监控
watch -n 1 nvidia-smi

# 或使用更详细的监控
nvidia-smi dmon -s u
```

### 9.3 API 性能测试

使用 Python 测试脚本：

```python
import requests
import time

# vLLM 测试
def test_vllm():
    url = "http://localhost:8000/v1/completions"
    data = {
        "model": "qwen2.5-7b",
        "prompt": "请用一句话介绍人工智能。",
        "max_tokens": 50
    }
    
    start = time.time()
    response = requests.post(url, json=data)
    elapsed = time.time() - start
    
    print(f"响应时间: {elapsed:.2f}秒")
    print(f"结果: {response.json()['choices'][0]['text']}")

# Xinference 测试
def test_xinference():
    url = "http://localhost:9997/v1/chat/completions"
    data = {
        "model": "<model-uid>",
        "messages": [{"role": "user", "content": "你好"}]
    }
    
    response = requests.post(url, json=data)
    print(response.json())

# Ollama 测试
def test_ollama():
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen2.5:7b",
        "prompt": "什么是深度学习？",
        "stream": False
    }
    
    response = requests.post(url, json=data)
    print(response.json()['response'])

if __name__ == "__main__":
    test_vllm()
```

### 9.4 负载测试工具

使用 `wrk` 或 `hey` 进行压力测试：

```bash
# 安装 hey
go install github.com/rakyll/hey@latest

# 并发测试
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","prompt":"测试","max_tokens":10}' \
  http://localhost:8000/v1/completions
```

---

## 10. 常见问题排查

### 10.1 Docker 相关问题

#### ❌ 问题：Docker Desktop 无法启动

**症状**：Docker Desktop 图标显示红色，提示"Docker Desktop starting..."卡住

**解决方案**：

```powershell
# 1. 完全退出 Docker Desktop

# 2. 清理 Docker 数据（谨慎操作）
wsl --shutdown
wsl --unregister docker-desktop
wsl --unregister docker-desktop-data

# 3. 重启 Docker Desktop
```

#### ❌ 问题：WSL2 集成未生效

**症状**：在 WSL2 中执行 `docker` 命令提示未找到

**解决方案**：

1. 打开 Docker Desktop Settings
2. Resources → WSL Integration
3. 启用目标发行版并重启 Docker

---

### 10.2 GPU 相关问题

#### ❌ 问题：容器中 `nvidia-smi` 失败

**症状**：

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**检查清单**：

```bash
# 1. Windows 驱动版本
# 在 PowerShell 中
nvidia-smi

# 2. WSL2 中 GPU 可见性
wsl
nvidia-smi

# 3. Docker 是否使用 WSL2 后端
docker info | grep -i wsl

# 4. NVIDIA Container Toolkit（通常不需要在 WSL2 中手动安装）
```

**解决步骤**：

1. 更新 Windows 上的 NVIDIA 驱动至最新版本
2. 确保驱动版本支持 WSL2（≥470.xx）
3. 重启 Windows 系统
4. 验证 WSL2 中 GPU 可见性

#### ❌ 问题：CUDA Out of Memory

**症状**：容器日志显示 `CUDA out of memory`

**解决方案**：

```bash
# 1. 减少最大序列长度
--max-model-len 4096

# 2. 降低 GPU 内存利用率
--gpu-memory-utilization 0.8

# 3. 减少并发请求数
--max-num-seqs 128

# 4. 使用量化模型
# 例如：使用 GPTQ/AWQ 量化版本

# 5. 启用 CPU offload（Xinference）
# 在 Web UI 中配置
```

---

### 10.3 网络与端口问题

#### ❌ 问题：无法访问 `localhost:<port>`

**症状**：浏览器或 curl 访问超时

**排查步骤**：

```bash
# 1. 确认容器正在运行
docker ps | grep <container-name>

# 2. 检查端口映射
docker port <container-name>

# 3. 测试容器内部连接
docker exec <container-name> curl http://localhost:8000/health

# 4. 检查 Windows 防火墙
# 在 PowerShell（管理员）中
New-NetFirewallRule -DisplayName "Allow Docker Ports" -Direction Inbound -LocalPort 8000,9997,11434 -Protocol TCP -Action Allow
```

#### ❌ 问题：端口被占用

**症状**：

```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**解决方案**：

```powershell
# 查找占用端口的进程
netstat -ano | findstr :8000

# 结束进程（记下 PID）
taskkill /PID <pid> /F

# 或更改容器端口映射
docker run -p 8001:8000 ...
```

---

### 10.4 模型加载问题

#### ❌ 问题：模型格式不兼容

**症状**：

```
ValueError: Model format 'safetensors' not supported
```

**解决方案**：

1. 确认框架支持的模型格式：
   - vLLM: PyTorch (`.bin`), SafeTensors (`.safetensors`)
   - Xinference: PyTorch, GGUF, GGML
   - Ollama: GGUF

2. 转换模型格式（如需要）：

```python
# PyTorch → SafeTensors
from transformers import AutoModel
model = AutoModel.from_pretrained("path/to/model")
model.save_pretrained("path/to/output", safe_serialization=True)
```

#### ❌ 问题：模型文件权限错误

**症状**：

```
PermissionError: [Errno 13] Permission denied: '/models/...'
```

**解决方案**：

```bash
# 在 WSL2 中修改权限
sudo chown -R $USER:$USER ~/models

# 或在 Docker 中指定用户
docker run --user $(id -u):$(id -g) ...
```

