"""轻量时空编码器 (CPU友好): TCN + 变量自注意力
复用第一篇 STMP-MIL 思路: 时间编码 + 变量描述子(均值/离散度/斜率) + 密集自注意力.
输出 z ∈ R^d (d=24, 第一篇同款)."""
import torch
import torch.nn as nn


class VariableDescriptors(nn.Module):
    """每变量的统计描述子 (第一篇 STMP-MIL 已用): 均值/离散度/斜率."""
    def forward(self, x):  # x: (B, n, l, D)
        mean = x.mean(dim=2)                  # (B,n,D)
        disp = x.std(dim=2)                   # (B,n,D)
        # 线性斜率: 对 l 维做最小二乘斜率
        l = x.shape[2]
        t = torch.arange(l, device=x.device, dtype=x.dtype).view(1, 1, l, 1)
        t_centered = t - t.mean()
        x_centered = x - x.mean(dim=2, keepdim=True)
        slope = (x_centered * t_centered).sum(dim=2) / (t_centered.pow(2).sum() + 1e-8)
        return torch.cat([mean, disp, slope], dim=-1)  # (B,n,3D)


class TemporalConvEncoder(nn.Module):
    """1D-TCN 编码器: instance (l,D) -> (d,)."""
    def __init__(self, D, channels=(32, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = D
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):  # x: (B*n, D, l)
        return self.net(x)  # (B*n, C, l)


class VariableSelfAttention(nn.Module):
    """密集变量自注意力 (第一篇 STMP-MIL 组件)."""
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):  # x: (N, D, d_model) 变量维度做注意力
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V  # (N, D, d_model)


class InstanceEncoder(nn.Module):
    """完整 instance 编码器 -> z ∈ R^d.
    pipeline: TCN(时间) + 变量描述子 + 变量自注意力 -> 池化 -> d_embed."""
    def __init__(self, D, d_embed=24, channels=(32, 64), kernel_size=3, dropout=0.1, use_var_attn=True):
        super().__init__()
        self.use_var_attn = use_var_attn
        self.tcn = TemporalConvEncoder(D, channels, kernel_size, dropout)
        self.desc = VariableDescriptors()
        # TCN 通道均值池化(d) + 描述子(3D) -> 拼接后投影
        fused_in = channels[-1] + 3 * D
        if use_var_attn:
            self.var_attn = VariableSelfAttention(channels[-1])
            fused_in = channels[-1] + 3 * D  # 变量注意力后仍池化回 d
        self.proj = nn.Sequential(
            nn.Linear(fused_in, 2 * d_embed), nn.ReLU(), nn.Linear(2 * d_embed, d_embed)
        )
        self.d_embed = d_embed

    def forward(self, x):  # x: (B, n, l, D)
        B, n, l, D = x.shape
        # TCN 时间编码
        flat = x.reshape(B * n, l, D).permute(0, 2, 1)  # (B*n, D, l)
        t = self.tcn(flat)                              # (B*n, C, l)
        t = t.mean(dim=2)                               # (B*n, C) 时间池化
        # 变量描述子
        d3 = self.desc(x).reshape(B * n, 3 * D)         # (B*n, 3D)
        fused = torch.cat([t, d3], dim=-1)              # (B*n, C+3D)
        z = self.proj(fused)                            # (B*n, d)
        return z.view(B, n, self.d_embed)
