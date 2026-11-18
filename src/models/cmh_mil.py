"""CMH-MIL 完整模型
encoder(时空) + 多球原型 + bag聚合 + (可选)多球归因(创新点③).
前向返回 instance分数/bag分数/嵌入, 供损失与评估."""
import torch
import torch.nn as nn
from .encoder import InstanceEncoder
from .hyperspheres import MultiHypersphereLayer


class CMHMIL(nn.Module):
    def __init__(self, D, d_embed=24, k=3, channels=(32, 64), kernel_size=3,
                 dropout=0.1, use_var_attn=False, pool="max", init_centers=None,
                 radius_mode="quantile", radius_q=0.9, radius_ema=0.9):
        super().__init__()
        self.encoder = InstanceEncoder(D, d_embed, channels, kernel_size, dropout, use_var_attn)
        self.hypers = MultiHypersphereLayer(d_embed, k, init_centers,
                                            radius_mode=radius_mode, radius_q=radius_q,
                                            radius_ema=radius_ema)
        self.pool = pool

    @torch.no_grad()
    def init_centers_kmeans(self, X_sample, device="cpu"):
        """用训练数据做 k-means 初始化球中心 (防随机初始化导致坍缩).
        X_sample: (N, n, l, D) 训练bag样本子集. 训练前调用."""
        from sklearn.cluster import KMeans
        self.encoder.eval()
        z = self.encoder(X_sample.to(device))  # (N,n,d)
        z_flat = z.reshape(-1, z.shape[-1]).cpu().numpy()
        km = KMeans(n_clusters=self.hypers.k, n_init=10, random_state=0).fit(z_flat)
        centers = torch.from_numpy(km.cluster_centers_).float().to(device)
        self.hypers.centers.data.copy_(centers)
        # 半径初始化为各簇内平均距离
        d2 = torch.cdist(centers, centers)
        R_init = d2[d2 > 0].median().item() / 2 if (d2 > 0).any() else 1.0
        self.hypers.radii.data.fill_(max(0.5, R_init))
        return centers

    def forward(self, x, return_z=False):
        """x: (B, n, l, D) -> instance_scores (B,n), bag_score (B,), z (B,n,d)."""
        z = self.encoder(x)                       # (B,n,d)
        B, n, d = z.shape
        z_flat = z.reshape(B * n, d)
        scores = self.hypers.instance_score(z_flat).view(B, n)  # (B,n)
        # bag 聚合 (⑦: 新增 max 分支 — any-point 语义, 与论文 Eq(8) 对齐)
        if self.pool == "softmax":
            w = torch.softmax(scores, dim=-1)
            bag_score = (w * scores).sum(dim=-1)
        elif self.pool == "max":
            bag_score = scores.max(dim=-1).values
        elif self.pool == "mean":
            bag_score = scores.mean(dim=-1)
        elif self.pool == "topk":
            kk = max(1, n // 4)
            bag_score = scores.topk(kk, dim=-1).values.mean(dim=-1)
        else:
            raise ValueError(self.pool)
        if return_z:
            return scores, bag_score, z
        return scores, bag_score

    def attribute(self, x):
        """多球语义归因 (创新点③): 异常->最近正常球偏离方向->变量级.
        x: (B,n,l,D) -> 归因 (B,n,D).
        注意: 归因需要梯度, 不能用 no_grad; 用 enable_grad 局部开启."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            z2 = self.encoder(x_req)  # (B,n,d)
            B, n, d = z2.shape
            centers = self.hypers.centers.detach()  # (k,d)
            # 最近球
            d2 = torch.cdist(z2.reshape(-1, d), centers)
            nn_idx = d2.argmin(dim=-1)  # (B*n,)
            delta = z2.reshape(-1, d) - centers[nn_idx]  # 偏离向量 (B*n,d)
            # z 对 x 的雅可比与偏离向量乘积 -> 输入空间归因
            g = torch.autograd.grad(
                z2, x_req, grad_outputs=delta.view(B, n, d),
                retain_graph=False, create_graph=False
            )[0]  # (B,n,l,D)
        # 对 instance 时间维取绝对值均值 -> 变量级
        attr = g.abs().mean(dim=2)  # (B,n,D)
        return attr.detach()
