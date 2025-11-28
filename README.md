# aicodeing

面向中文开发者的大模型学习与实战工程。在 Windows 环境下从零搭建、验证到运行推理示例与练习,提供循序渐进的学习路径与可运行代码。

如果本项目对你有帮助,欢迎为仓库点亮 Star。

## 特性

- 适配 Windows 10/11 与 4GB 显存显卡
- 提供完整环境搭建指南与验证清单
- 可直接运行的示例与实验练习
- 采用 MIT 开源协议,自由使用与二次开发

## 目录结构

- `大模型学习体系/00-基础与环境/README.md` 环境搭建与实践指南
- `大模型学习体系/00-基础与环境/examples/` 可运行示例
  - `test_pytorch.py` 基础环境与 CUDA 验证脚本
  - `tinyllama_infer.py` TinyLlama 推理示例
  - `qwen_chat.py` Qwen2.5 对话示例
  - `llm_demo.py` 交互式推理脚本版示例
- `大模型学习体系/00-基础与环境/labs/` 练习脚本
  - `lab01_env_check.py` 环境综合检查
  - `lab02_pytorch_basics.py` PyTorch 基础练习
  - `lab03_tinyllama_lab.py` TinyLlama 实战练习
  - `lab04_qwen_lab.py` Qwen2.5 实战练习

示例入口参考:

- `大模型学习体系/00-基础与环境/examples/test_pytorch.py:1`
- `大模型学习体系/00-基础与环境/examples/tinyllama_infer.py:1`
- `大模型学习体系/00-基础与环境/examples/qwen_chat.py:1`
- `大模型学习体系/00-基础与环境/examples/llm_demo.py:1`

## 快速开始

- 按照 `大模型学习体系/00-基础与环境/README.md` 完成环境搭建
- 激活虚拟环境后运行示例

示例运行:

```cmd
python 大模型学习体系/00-基础与环境/examples/test_pytorch.py
python 大模型学习体系/00-基础与环境/examples/tinyllama_infer.py
python 大模型学习体系/00-基础与环境/examples/qwen_chat.py
```

使用 `uv`:

```cmd
uv run python 大模型学习体系/00-基础与环境/examples/test_pytorch.py
```

## 未来学习路径

- 基础环境
  - 了解 Python/Conda/uv 与 VSCode/Jupyter 的使用
  - 完成 PyTorch 与 Transformers 的安装验证
- 推理实践
  - 跑通 TinyLlama 与 Qwen2.5 的基础推理与对话
  - 理解采样参数(`temperature`,`top_p`,`max_new_tokens`)的影响
- 模型微调
  - 学习 LoRA/QLoRA 等参数高效微调技术
  - 选择中文数据集进行指令微调与评估
- 应用开发
  - 构建命令行/桌面/Web 聊天应用
  - 集成检索增强(RAG)与工具调用
- 部署与优化
  - 使用量化与图优化降低成本与提升性能
  - 尝试本地部署与轻量云部署
- 进阶研究
  - 跟进开源生态与论文,探索不同模型结构与推理优化

推荐资源:

- PyTorch 官方教程: https://pytorch.org/tutorials/
- Hugging Face 文档: https://huggingface.co/docs
- Transformers 课程: https://huggingface.co/course

## 贡献

- 欢迎通过 Issue 或 PR 提交改进建议与示例
- 推荐遵循 Windows 优先的可复现原则,并附运行说明

## 协议

- 本项目使用 MIT 协议,详见 `LICENSE`

## Star 一下

- 如果你觉得本项目有帮助,请为仓库点亮 Star,这将鼓励我们持续完善示例与学习路径