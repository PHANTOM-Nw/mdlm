#!/usr/bin/env python3
"""
最小化的MDLM训练循环
对比DDPM训练，帮助理解完整流程
"""
import torch
import torch.nn.functional as F

class MinimalMDLM:
    """
    简化的MDLM，去除所有工程细节，保留核心逻辑
    对比DDPM的训练循环
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.mask_index = vocab_size  # MASK token

    def noise_schedule(self, t):
        """
        类比DDPM: beta_schedule(t)
        返回: move_chance - token被MASK的概率
        """
        # 简化的log-linear schedule
        sigma = -torch.log1p(-(1 - 1e-3) * t)
        move_chance = 1 - torch.exp(-sigma)
        return move_chance, sigma

    def q_sample(self, x0, t):
        """
        前向加噪: q(xt | x0)
        类比DDPM: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """
        move_chance, sigma = self.noise_schedule(t)

        # 按概率将tokens替换为MASK
        # 对应 diffusion.py:575-586
        move_indices = torch.rand_like(x0.float()) < move_chance
        xt = torch.where(move_indices, self.mask_index, x0)
        return xt, sigma

    def model_forward(self, xt, sigma):
        """
        模拟神经网络预测（实际是DIT/DiMamba）
        输入: xt (带噪声的tokens), sigma (时间步)
        输出: logits (每个位置每个token的得分)
        """
        batch_size, seq_len = xt.shape
        # 这里用随机logits模拟，实际是复杂的Transformer
        logits = torch.randn(batch_size, seq_len, self.vocab_size + 1)
        return logits

    def subs_parameterization(self, logits, xt):
        """
        SUBS参数化 - 核心创新
        对应 diffusion.py:261-277
        """
        neg_infinity = -1000000.0

        # 步骤1: MASK位置的logit设为-∞
        logits[:, :, self.mask_index] = neg_infinity

        # 步骤2: Log softmax归一化
        log_probs = F.log_softmax(logits, dim=-1)

        # 步骤3: 未MASK位置设为确定值
        unmasked = (xt != self.mask_index)
        log_probs[unmasked] = neg_infinity
        log_probs[unmasked, xt[unmasked]] = 0

        return log_probs

    def compute_loss(self, x0, log_probs, sigma):
        """
        计算损失
        对应 diffusion.py:883-894
        """
        # 提取x0位置的log概率
        log_p_theta = log_probs.gather(
            dim=-1,
            index=x0.unsqueeze(-1)
        ).squeeze(-1)

        # 带权重的NLL损失（简化版）
        loss = -log_p_theta.mean()
        return loss

    def training_step(self, x0):
        """
        完整的训练步骤
        对应 diffusion.py:847-894的_forward_pass_diffusion
        """
        batch_size = x0.shape[0]

        # 1. 采样随机时间步 t ~ Uniform(0, 1)
        t = torch.rand(batch_size)

        # 2. 前向加噪: q(xt | x0)
        xt, sigma = self.q_sample(x0, t)

        # 3. 神经网络预测
        logits = self.model_forward(xt, sigma)

        # 4. SUBS参数化
        log_probs = self.subs_parameterization(logits, xt)

        # 5. 计算损失
        loss = self.compute_loss(x0, log_probs, sigma)

        return loss, xt

    def sampling_step(self, x, t, dt):
        """
        采样一步: p(x_{t-dt} | x_t)
        类比DDPM的反向去噪步骤
        对应 diffusion.py:592-610的_ddpm_caching_update
        """
        sigma_t, _ = self.noise_schedule(t)
        sigma_s, _ = self.noise_schedule(t - dt)

        # 1. 神经网络预测
        logits = self.model_forward(x, sigma_t)
        log_probs = self.subs_parameterization(logits, x)
        p_x0 = log_probs.exp()

        # 2. 计算转移概率 q(x_{t-dt} | x_t, x_0)
        move_chance_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
        move_chance_s = sigma_s.unsqueeze(-1).unsqueeze(-1)

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s.squeeze(-1)

        # 3. 采样新的x
        x_new = torch.distributions.Categorical(probs=q_xs).sample()

        # 4. 保持未MASK的位置不变
        mask_positions = (x == self.mask_index)
        x_new = torch.where(mask_positions, x_new, x)

        return x_new

def demo_training():
    """
    演示训练流程
    """
    print("="*70)
    print("MDLM最小化训练演示")
    print("="*70)

    mdlm = MinimalMDLM(vocab_size=1000)

    # 模拟一个batch的数据
    batch_size, seq_len = 2, 8
    x0 = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"\n输入数据 x0 (干净tokens):")
    print(f"  形状: {x0.shape}")
    print(f"  内容: {x0.tolist()}\n")

    # 训练一步
    print("执行训练步骤...")
    loss, xt = mdlm.training_step(x0)

    print(f"\n加噪后 xt (部分被MASK):")
    print(f"  内容: {xt.tolist()}")
    print(f"  MASK位置: {(xt == mdlm.mask_index).sum().item()} / {xt.numel()}")
    print(f"\n损失: {loss.item():.4f}")

    print("\n" + "="*70)
    print("与DDPM对比:")
    print("  DDPM训练步骤:")
    print("    1. t = random(0,1)")
    print("    2. xt = sqrt(α̅t)*x0 + sqrt(1-α̅t)*ε  (加高斯噪声)")
    print("    3. ε_pred = UNet(xt, t)")
    print("    4. loss = MSE(ε, ε_pred)")
    print("")
    print("  MDLM训练步骤:")
    print("    1. t = random(0,1)")
    print("    2. xt = mask_tokens(x0, p=move_chance(t))  (替换为MASK)")
    print("    3. log_p = DIT(xt, t)")
    print("    4. loss = -log p(x0|xt)")
    print("="*70)

def demo_sampling():
    """
    演示采样流程
    """
    print("\n" + "="*70)
    print("MDLM采样演示")
    print("="*70)

    mdlm = MinimalMDLM(vocab_size=1000)

    # 从纯噪声开始（全MASK）
    batch_size, seq_len = 1, 8
    x = torch.full((batch_size, seq_len), mdlm.mask_index, dtype=torch.long)

    print(f"\n初始状态 x (全MASK):")
    print(f"  {x.tolist()}\n")

    # 逐步去噪
    num_steps = 5
    dt = 1.0 / num_steps

    print("逐步去噪过程:")
    for i in range(num_steps):
        t = torch.tensor([1.0 - i * dt])
        x = mdlm.sampling_step(x, t, dt)

        num_masks = (x == mdlm.mask_index).sum().item()
        print(f"  步骤 {i+1}/5, t={t.item():.2f}, 剩余MASK: {num_masks}/{seq_len}")
        print(f"    当前 x: {x.tolist()}")

    print("\n最终生成的tokens:")
    print(f"  {x.tolist()}")

    print("\n" + "="*70)
    print("与DDPM对比:")
    print("  DDPM采样: x_T (高斯噪声) → x_0 (清晰图像)")
    print("  MDLM采样: x_T (全MASK) → x_0 (完整文本)")
    print("="*70)

if __name__ == "__main__":
    demo_training()
    demo_sampling()

    print("\n" + "="*70)
    print("关键理解:")
    print("  1. MDLM是'离散版本的DDPM'")
    print("  2. MASK token = 高斯噪声")
    print("  3. 预测原始token = 预测噪声（但形式不同）")
    print("  4. SUBS简化了预测目标，本质是MLM")
    print("="*70)
