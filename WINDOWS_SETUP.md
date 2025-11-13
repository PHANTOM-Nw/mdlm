# Windows环境配置指南

本指南适用于在Windows平台上配置MDLM项目。

## 前置要求

- Python 3.9+
- NVIDIA GPU with CUDA 12.6+ support
- Git

## 安装步骤

### 1. 创建Python虚拟环境

```bash
# 使用venv创建虚拟环境
python -m venv mdlm-env

# 激活虚拟环境
mdlm-env\Scripts\activate
```

或使用conda:

```bash
conda create -n mdlm python=3.9
conda activate mdlm
```

### 2. 安装PyTorch (CUDA 12.6)

**重要**: 必须先安装PyTorch，再安装其他依赖。

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. 安装其他依赖

```bash
pip install -r requirements-windows.txt
```

### 4. 创建必要的目录

```bash
mkdir outputs
mkdir watch_folder
```

## Windows特定注意事项

### Flash Attention
Flash Attention在Windows上可能需要特殊处理：

**选项1**: 尝试直接安装（可能失败）
```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

**选项2**: 如果安装失败，可以跳过flash-attn，项目应该能以降级模式运行
```bash
pip install -r requirements-windows.txt --ignore-installed flash-attn
```

然后手动从requirements-windows.txt中移除flash-attn行后重新安装。

### Triton
- 使用 `triton-windows` 替代标准的 `triton` 包
- 确保已在requirements-windows.txt中配置

### Mamba-SSM
如果 `mamba-ssm` 安装失败：
```bash
# 可能需要安装Visual Studio Build Tools
# 下载地址: https://visualstudio.microsoft.com/downloads/
```

## 验证安装

运行以下命令验证环境配置：

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

预期输出应显示：
- PyTorch版本
- CUDA Available: True
- CUDA Version: 12.6或相近版本

## 运行示例

### 下载并测试预训练模型

```bash
python main.py mode=sample_eval eval.checkpoint_path=kuleshov-group/mdlm-owt data=openwebtext-split model.length=1024 sampling.predictor=ddpm_cache sampling.steps=1000 loader.eval_batch_size=1 sampling.num_sample_batches=1 backbone=hf_dit
```

## 常见问题

### 1. CUDA out of memory
- 减小 `loader.eval_batch_size`
- 减小 `model.length`

### 2. 找不到CUDA
- 确认NVIDIA驱动已安装
- 确认PyTorch安装时选择了正确的CUDA版本

### 3. 包编译失败
- 安装 Visual Studio Build Tools
- 确保有足够的磁盘空间

## 与Linux版本的差异

| 组件 | Linux (requirements.yaml) | Windows (requirements-windows.txt) |
|------|--------------------------|-------------------------------------|
| 环境管理 | Conda | pip/venv |
| PyTorch安装 | conda (cuda 12.1) | pip (cuda 12.6) |
| Triton | triton==2.2.0 | triton-windows |
| Flash Attention | flash-attn==2.5.6 | 可选 (可能不兼容) |

## 获取帮助

如遇到问题，请参考：
- [项目主页](https://github.com/s-sahoo/mdlm)
- [PyTorch官方文档](https://pytorch.org/get-started/locally/)
- [CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
