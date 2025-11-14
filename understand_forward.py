#!/usr/bin/env python3
"""
可视化MDLM的前向加噪过程
类比：DDPM中逐步加高斯噪声 → MDLM中逐步替换为MASK
"""
import torch
from transformers import AutoTokenizer
import sys
sys.path.append('/home/user/mdlm')

def visualize_forward_process():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text)

    print("="*60)
    print("MDLM前向加噪过程可视化")
    print("="*60)
    print(f"\n原始文本: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Token数量: {len(tokens)}")

    # MASK token
    mask_token_id = tokenizer.vocab_size  # 50257 for GPT-2
    tokenizer.add_special_tokens({'mask_token': '<MASK>'})

    print(f"\n{'时间步 t':<10} {'MASK比例':<15} {'加噪后的tokens'}")
    print("-"*60)

    # 模拟不同时间步的加噪
    torch.manual_seed(42)  # 固定随机种子便于理解
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        move_chance = t  # 简化版：实际使用 move_chance = 1 - exp(-sigma(t))

        # 按概率替换为MASK（对应diffusion.py:575-586的q_xt函数）
        noisy_tokens = []
        for tok in tokens:
            if torch.rand(1).item() < move_chance:
                noisy_tokens.append(mask_token_id)
            else:
                noisy_tokens.append(tok)

        num_masked = sum(1 for tok in noisy_tokens if tok == mask_token_id)
        mask_ratio = f"{num_masked}/{len(tokens)}"

        # 可视化
        token_strs = []
        for tok in noisy_tokens:
            if tok == mask_token_id:
                token_strs.append("[MASK]")
            else:
                token_strs.append(tokenizer.decode([tok]))

        print(f"t={t:<7.1f} {mask_ratio:<15} {' '.join(token_strs)}")

    print("\n" + "="*60)
    print("关键理解:")
    print("  - t=0.0: 完全干净（无MASK）")
    print("  - t=1.0: 完全噪声（全是MASK）")
    print("  - 类比DDPM: MASK token ≈ 高斯噪声")
    print("="*60)

if __name__ == "__main__":
    visualize_forward_process()
