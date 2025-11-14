#!/usr/bin/env python3
"""
理解SUBS参数化的核心创新
对比D3PM和SUBS的预测目标差异
"""
import torch
import torch.nn.functional as F

def compare_parameterizations():
    """
    对比三种参数化方法的预测目标
    """
    print("="*70)
    print("MDLM三种参数化方法对比")
    print("="*70)

    # 模拟场景
    vocab_size = 10
    seq_len = 5
    batch_size = 1
    mask_index = vocab_size - 1  # 9是MASK

    # 原始干净序列
    x0 = torch.tensor([[1, 2, 3, 4, 5]])  # 干净的tokens
    print(f"\n原始序列 x0: {x0.tolist()}")

    # 加噪后的序列（部分被MASK）
    xt = torch.tensor([[9, 2, 9, 4, 9]])  # 9代表MASK
    print(f"加噪序列 xt: {xt.tolist()}")
    print(f"被MASK的位置: [0, 2, 4]\n")

    # 模拟神经网络的原始输出（未归一化的logits）
    logits = torch.randn(batch_size, seq_len, vocab_size)
    print(f"神经网络原始logits形状: {logits.shape}")
    print("-"*70)

    # ===== 方法1: D3PM参数化 =====
    print("\n[方法1] D3PM参数化 (baseline)")
    print("  预测目标: 每个位置的完整token分布")

    d3pm_output = logits.clone()
    d3pm_output = F.log_softmax(d3pm_output, dim=-1)

    print(f"  输出形状: {d3pm_output.shape}")
    print(f"  含义: 对每个位置，预测vocab_size个token的概率")
    print(f"  例如位置0的预测分布:")
    print(f"    {d3pm_output[0, 0, :].exp().tolist()}")
    print(f"    (所有vocab的概率，包括MASK位置)")

    # ===== 方法2: SUBS参数化（本文提出）=====
    print("\n[方法2] SUBS参数化 (本文创新) - diffusion.py:261-277")
    print("  预测目标: 只预测原始token，非MASK位置设为确定值")

    subs_output = logits.clone()

    # 步骤1: MASK位置的logit设为-∞
    neg_infinity = -1000000.0
    subs_output[:, :, mask_index] = neg_infinity

    # 步骤2: 归一化（log softmax）
    subs_output = F.log_softmax(subs_output, dim=-1)

    # 步骤3: 对于未被MASK的位置，只保留原始token的概率
    unmasked_indices = (xt != mask_index)  # [False, True, False, True, False]
    subs_output[unmasked_indices] = neg_infinity  # 全部设为-∞
    # 然后把对应原始token的位置设为0（概率=1）
    subs_output[unmasked_indices, xt[unmasked_indices]] = 0

    print(f"  输出形状: {subs_output.shape}")
    print(f"  位置0 (被MASK): 预测分布 = {subs_output[0, 0, :5].tolist()}")
    print(f"  位置1 (未MASK,原值=2): 预测 = one-hot at 2")
    print(f"    {subs_output[0, 1, :5].tolist()}")
    print(f"  关键: 未MASK位置直接用真实token，无需预测！")

    # ===== 方法3: SEDD参数化 =====
    print("\n[方法3] SEDD参数化 (score-based)")
    print("  预测目标: score函数 ∇log p(x_t|x_0)")
    print("  (更复杂，涉及score matching)")

    print("\n" + "="*70)
    print("核心优势对比:")
    print("  D3PM: 需要预测所有位置×所有vocab → 计算量大")
    print("  SUBS: 只预测MASK位置，其他位置free → 简化为MLM！")
    print("  SEDD: 基于分数，理论优雅但实现复杂")
    print("="*70)

    # ===== 损失函数对比 =====
    print("\n损失函数对比:")

    # SUBS损失（简化版）
    log_p_theta = subs_output.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    subs_loss = -log_p_theta.mean()
    print(f"  SUBS loss = -log p_θ(x0|xt) = {subs_loss.item():.4f}")
    print(f"  本质: 交叉熵损失，只在MASK位置计算")

    # D3PM损失（简化版）
    d3pm_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        x0.view(-1)
    )
    print(f"  D3PM loss = CE(logits, x0) = {d3pm_loss.item():.4f}")
    print(f"  本质: 标准交叉熵，所有位置计算")

    print("\n关键理解:")
    print("  SUBS把复杂的扩散损失简化成了BERT-style的MLM损失！")
    print("  这就是论文标题'Simple and Effective'的含义。")
    print("="*70)

if __name__ == "__main__":
    compare_parameterizations()
