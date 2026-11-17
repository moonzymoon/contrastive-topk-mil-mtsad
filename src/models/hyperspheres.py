"""多球原型层 (W2-⑤b2 重建版, 2026-11-17 总管裁定)
k 个超球 { (c_j, R_j) }. 硬分配 m=argmin||z-c||.
半径两种模式:
  learnable (v2历史): R_j 为可学习参数 — 已证实目标无下界(∂L/∂R≈-1.98R), 保留仅供复现
  quantile  (b2, v3默认): R_j = EMA(Quantile_q{d(z,c_j): z assigned to j}), q默认0.9
    - 无梯度通道: 半径是统计量非参数, 病态通道结构性消失
    - 分数语义: s = d² - Q90(d)² = 到最近中心的距离是否超出正常90%分位 (conformal同构)
    - q 可参数化 (P90/P95/P99 敏感性分析, W4)
实例分数: 最近球距离-半径² (s>0 落球外=异常倾向)."""
import torch
import torch.nn as nn


class MultiHypersphereLayer(nn.Module):
    def __init__(self, d_embed=24, k=3, init_centers=None,
                 radius_mode="quantile", radius_q=0.9, radius_ema=0.9):
        super().__init__()
        self.k = k
        self.radius_mode = radius_mode
        self.radius_q = radius_q
        self.radius_ema = radius_ema
        # 中心 c_j
        if init_centers is not None:
            self.centers = nn.Parameter(init_centers.clone().detach())
        else:
            self.centers = nn.Parameter(torch.randn(k, d_embed) * 0.01)
        if radius_mode == "learnable":
            self.radii = nn.Parameter(torch.ones(k) * 1.0)   # v2 行为(历史)
        elif radius_mode == "quantile":
            # buffer: 不进 optimizer, 无梯度 (b2 核心)
            self.register_buffer("radii", torch.ones(k) * 1.0)
        else:
            raise ValueError(radius_mode)

    def hard_assign(self, z):  # z: (..., d) -> 分配索引 (...)
        d2 = torch.cdist(z, self.centers).pow(2)  # (..., k)
        return d2.argmin(dim=-1)                  # (...)

    @torch.no_grad()
    def update_radii(self, z):
        """b2: EMA 分位数半径更新. 训练每步(或每epoch)调用.
        R_j ← m·R_j + (1-m)·Quantile_q{sqrt(d²(z_i,c_j)): i assigned to j}
        球内样本不足时保持原值 (不更新)."""
        if self.radius_mode != "quantile":
            return
        d = torch.cdist(z, self.centers)              # (N, k)
        assign = d.argmin(dim=-1)                     # (N,)
        for j in range(self.k):
            members = d[assign == j, j]
            if len(members) < 8:                       # 样本不足, 跳过防噪声
                continue
            q_val = torch.quantile(members, self.radius_q)
            self.radii[j] = (self.radius_ema * self.radii[j]
                             + (1 - self.radius_ema) * q_val)

    def instance_score(self, z):  # z: (..., d) -> s (...)
        """最近球距离 - 半径^2. s>0 = 落球外."""
        d2 = torch.cdist(z, self.centers).pow(2)  # (..., k)
        R2 = self.radii.pow(2)                     # (k,)
        return (d2 - R2).min(dim=-1).values        # (...) 最近球

    def soft_score(self, z, tau_s=1.0):  # 加权形式(消融用)
        d2 = torch.cdist(z, self.centers).pow(2)
        R2 = self.radii.pow(2)
        w = torch.softmax(-d2 / tau_s, dim=-1)     # 软归属
        return (w * (d2 - R2)).sum(dim=-1)

    def collapse_diagnostic(self, z):
        """坍缩诊断: 返回每球样本数 + 球间最小中心距 (阴性结果留存)."""
        with torch.no_grad():
            assign = self.hard_assign(z)           # (N,)
            counts = torch.bincount(assign, minlength=self.k)
            cdist = torch.cdist(self.centers, self.centers)
            cdist.fill_diagonal_(float("inf"))
            min_sep = cdist.min().item()
        return counts.tolist(), min_sep
