from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class Critic(nn.Module):
    """Set Transformer Critic for CTDE.

    Input: Set of agent embeddings [Batch, m, F] (m varies, handled by masking/padding or pure set op)
    Output: Global Value V(s)

    The Set Transformer (ICML 2019) is permutation invariant and handles variable set sizes naturally.
    We use SAB (Self-Attention Block) for encoding and PMA (Pooling by Multihead Attention) for aggregation.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 1,
                 num_heads: int = 4, num_inds: int = 16):
        super(Critic, self).__init__()
        self.enc = nn.Sequential(
            ISAB(input_dim, hidden_dim, num_heads, num_inds, ln=True),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=True)
        )
        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, 1, ln=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [batch, n_agents, feature_dim]
        # We assume X is already padded if batch > 1, but for typical rollout batch=1 it's just [1, m, F]
        # If m varies inside a batch, we'd need a mask, but here we process episode-steps which usually have uniform m per batch or batch=1.
        h = self.enc(X)
        out = self.dec(h).squeeze(-1).squeeze(-1) # [batch]
        return out


class PopArtCritic(nn.Module):
    """带有PopArt归一化的Critic包装器
    
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
    
    === 数学原理 ===
    将Critic的输出分解为: V(s) = σ * Ṽ(s) + μ
    其中:
    - μ, σ 是return分布的running mean和std
    - Ṽ(s) 是归一化后的预测
    
    当return分布变化时，只更新 μ, σ，同时调整最后一层权重以保持输出连续性。
    这解决了非平稳目标函数导致的Critic学习不稳定问题。
    
    参考: DeepMind "Multi-task RL with PopArt" (2018)
    """
    
    def __init__(self, base_critic: Critic, beta: float = 0.0003):
        super(PopArtCritic, self).__init__()
        self.critic = base_critic
        self.beta = beta  # Running mean/std的更新率 (EMA decay)
        
        # PopArt统计量 (使用buffer确保保存和加载时正确处理)
        self.register_buffer('mu', torch.tensor(0.0))        # Running mean
        self.register_buffer('sigma', torch.tensor(1.0))     # Running std
        self.register_buffer('nu', torch.tensor(1.0))        # Running second moment (E[x^2])
        self.register_buffer('_initialized', torch.tensor(False))  # 是否已初始化
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播，返回反归一化后的实际值预测"""
        # 获取归一化的预测 Ṽ(s)
        v_normalized = self.critic(X)
        # 反归一化得到实际值预测: V(s) = σ * Ṽ(s) + μ
        v_actual = self.sigma * v_normalized + self.mu
        return v_actual
    
    def update_stats(self, returns: torch.Tensor):
        """使用新的return batch更新统计量，并调整输出层权重
        
        === 关键数学 ===
        要保持输出不变，需要:
        V_new(s) = σ_new * Ṽ_new(s) + μ_new = V_old(s) = σ_old * Ṽ_old(s) + μ_old
        
        设 Ṽ(s) = W*h + b (最后一层线性变换)
        则:
        W_new = W_old * σ_old / σ_new
        b_new = (σ_old * b_old + μ_old - μ_new) / σ_new
        """
        with torch.no_grad():
            # 计算batch统计量
            batch_mean = returns.mean()
            batch_var = returns.var()
            batch_second_moment = batch_var + batch_mean ** 2
            
            # 首次初始化
            if not self._initialized.item():
                self.mu.copy_(batch_mean)
                self.nu.copy_(batch_second_moment)
                self.sigma.copy_(torch.sqrt(batch_var).clamp(min=1e-4))
                self._initialized.fill_(True)
                return
            
            # 保存旧的统计量
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()
            
            # EMA更新统计量
            new_mu = (1 - self.beta) * self.mu + self.beta * batch_mean
            new_nu = (1 - self.beta) * self.nu + self.beta * batch_second_moment
            new_sigma = torch.sqrt((new_nu - new_mu ** 2).clamp(min=1e-8)).clamp(min=1e-4)
            
            # === 调整最后一层权重以保持输出连续性 ===
            # 获取输出层 (dec的最后一个Linear)
            output_layer = self.critic.dec[-1]
            if isinstance(output_layer, nn.Linear):
                # W_new = W_old * σ_old / σ_new
                scale_factor = old_sigma / new_sigma
                output_layer.weight.data.mul_(scale_factor)
                
                # b_new = (σ_old * b_old + μ_old - μ_new) / σ_new
                output_layer.bias.data = (old_sigma * output_layer.bias.data + old_mu - new_mu) / new_sigma
            
            # 更新统计量buffer
            self.mu.copy_(new_mu)
            self.sigma.copy_(new_sigma)
            self.nu.copy_(new_nu)
    
    def normalize_targets(self, returns: torch.Tensor) -> torch.Tensor:
        """将return归一化为训练目标
        
        G_normalized = (G - μ) / σ
        """
        return (returns - self.mu) / self.sigma
    
    def get_stats(self) -> dict:
        """返回当前统计量，用于日志记录"""
        return {
            'popart_mu': float(self.mu.item()),
            'popart_sigma': float(self.sigma.item()),
        }

