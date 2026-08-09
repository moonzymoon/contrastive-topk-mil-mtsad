"""多球原型层 (创新点①, 对接导师 C2 多球 SVDD)
k 个超球 { (c_j, R_j) }. 硬分配 m=argmin||z-c|| (导师C2). 球样本数约束防坍缩.
实例分数: 最近球距离 (s>0 落球外=异常倾向)."""
import torch
import torch.nn as nn


class MultiHypersphereLayer(nn.Module):
    def __init__(self, d_embed=24, k=3, init_centers=None):
        super().__init__()
        self.k = k
        # 中心 c_j (导师C2: VAE预训练初始化, 否则随机)
        if init_centers is not None:
            self.centers = nn.Parameter(init_centers.clone().detach())
        else:
            self.centers = nn.Parameter(torch.randn(k, d_embed) * 0.01)
        # 半径 R_j (可学习, 初始为正)
        self.radii = nn.Parameter(torch.ones(k) * 1.0)

    def hard_assign(self, z):  # z: (..., d) -> 分配索引 (...)
        d2 = torch.cdist(z, self.centers).pow(2)  # (..., k)
        return d2.argmin(dim=-1)                  # (...)

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
