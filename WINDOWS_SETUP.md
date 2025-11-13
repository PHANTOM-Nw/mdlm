# Windows环境配置指南

本指南适用于在Windows平台上配置MDLM项目。

## 🚀 快速开始（推荐）

如果您遇到 `causal-conv1d` 编译错误，使用这个简化流程：

```bash
# 1. 创建并激活虚拟环境
conda create -n mdlm python=3.9
conda activate mdlm

# 2. 安装PyTorch (CUDA 12.6)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安装核心依赖（跳过问题包）
pip install -r requirements-windows-minimal.txt

# 4. 创建目录
mkdir outputs
mkdir watch_folder

# 5. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

完成！现在可以运行项目了（使用 DiT 架构）。

---

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

**推荐方法A: 使用最小化依赖（最可靠）** ⭐
```bash
# 只安装核心依赖，跳过可能失败的包
pip install -r requirements-windows-minimal.txt
```

**方法B: 完整安装但跳过问题包**
```bash
# requirements-windows.txt已将问题包注释掉
pip install -r requirements-windows.txt
```

**方法C: 尝试全部安装（可能部分失败）**
```bash
# 先安装基础依赖（不包括编译问题的包）
pip install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 jupyter==1.0.0 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 wandb==0.13.5

# 安装Triton for Windows
pip install triton-windows

# 尝试安装causal-conv1d (通常会失败)
pip install causal-conv1d==1.1.3.post1

# 尝试安装mamba-ssm (通常会失败)
pip install mamba-ssm==1.1.4

# 尝试安装flash-attn (通常会失败)
pip install flash-attn==2.5.6 --no-build-isolation
```

### 4. 创建必要的目录

```bash
mkdir outputs
mkdir watch_folder
```

## Windows特定注意事项与问题解决

### ⚠️ causal-conv1d 编译失败

**问题**: `causal-conv1d` 在Windows上从源码编译失败，错误：`fatal error C1083: 无法打开源文件`

**解决方案**:

**方案1: 跳过causal-conv1d（推荐）**
```bash
# causal-conv1d主要用于Mamba模型，如果不使用Mamba架构，可以跳过
# 项目默认使用DiT (Diffusion Transformer)架构，不依赖此包
```
从 `requirements-windows.txt` 中移除或注释掉 `causal-conv1d==1.1.3.post1` 行。

**方案2: 使用预编译wheel（如果可用）**
```bash
# 检查是否有适合您CUDA版本的预编译wheel
# 访问: https://github.com/Dao-AILab/causal-conv1d/releases
pip install causal-conv1d --find-links https://github.com/Dao-AILab/causal-conv1d/releases
```

**方案3: 安装编译工具（高级）**
如果必须使用Mamba模型：
1. 安装 Visual Studio 2022 Community (含C++工具)
2. 安装 Ninja 构建系统: `pip install ninja`
3. 确保 CUDA Toolkit 完整安装
4. 重新尝试安装

### ⚠️ Mamba-SSM 依赖问题

**问题**: `mamba-ssm` 依赖 `causal-conv1d`，如果后者安装失败，前者也会失败。

**解决方案**:
- 如果不使用Mamba架构（使用DiT），可以跳过此包
- 从 `requirements-windows.txt` 中移除或注释掉 `mamba-ssm==1.1.4` 行

```bash
# 在配置文件中使用 backbone=dit 而不是 backbone=mamba
```

### ⚠️ Flash Attention

**问题**: Flash Attention在Windows上编译困难

**解决方案**:

**方案1**: 尝试预编译版本
```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

**方案2**: 跳过（推荐）
```bash
# 项目可以在没有flash-attn的情况下运行，只是速度稍慢
# 从requirements-windows.txt中移除此行
```

### ✅ Triton
- 使用 `triton-windows` 替代标准的 `triton` 包
- 已在requirements-windows.txt中正确配置

### 📋 最小化安装清单（保证核心功能）

如果遇到多个编译问题，可以使用最小化依赖集：

```bash
# 必需的核心依赖
pip install datasets==2.18.0
pip install einops==0.7.0
pip install fsspec==2024.2.0
pip install h5py==3.10.0
pip install hydra-core==1.3.2
pip install lightning==2.2.1
pip install notebook==7.1.1
pip install omegaconf==2.3.0
pip install pandas==2.2.1
pip install transformers==4.38.2
pip install wandb==0.13.5
pip install timm==0.9.16
pip install triton-windows

# 可选但推荐
pip install rich==13.7.1
pip install seaborn==0.13.2
pip install scikit-learn==1.4.0
```

这样可以使用DiT架构运行MDLM的核心功能。

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

### 1. causal-conv1d / mamba-ssm 编译失败
**问题**: `fatal error C1083: 无法打开源文件`

**解决方案**:
- **推荐**: 跳过这些包，使用DiT架构（默认）
- 从 `requirements-windows.txt` 移除以下行：
  ```
  causal-conv1d==1.1.3.post1
  mamba-ssm==1.1.4
  ```
- 确保使用 `backbone=dit` 或 `backbone=hf_dit` (不使用 `backbone=mamba`)

### 2. CUDA out of memory
- 减小 `loader.eval_batch_size`
- 减小 `model.length`

### 3. 找不到CUDA
- 确认NVIDIA驱动已安装
- 确认PyTorch安装时选择了正确的CUDA版本
- 运行 `nvidia-smi` 检查GPU状态

### 4. 缺少 Visual Studio Build Tools
如果需要编译包（不推荐新手）：
- 下载 Visual Studio 2022 Community
- 安装时选择 "使用C++的桌面开发" 工作负载
- 下载地址: https://visualstudio.microsoft.com/downloads/

### 5. pip install 超时
```bash
pip install --default-timeout=100 -r requirements-windows.txt
```

### 6. 导入错误: No module named 'causal_conv1d'
**原因**: 代码尝试导入但包未安装

**临时解决**: 修改代码跳过Mamba相关导入，或确保不使用Mamba架构

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
