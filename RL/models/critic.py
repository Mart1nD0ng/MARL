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

