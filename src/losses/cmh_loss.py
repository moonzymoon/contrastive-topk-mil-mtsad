"""CMH-MIL 多目标损失
= bag_BCE + λ1·compact + λ2·sep + λ3·rank + α·ELBO-MoG(导师C2)
权重沿用第一篇已验证组合 (compact 0.20 / sep 0.02 / rank 1.0)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bag_bce_loss(instance_scores, y_bag, pool="softmax"):
    """bag 分数聚合 -> bag BCE.
    instance_scores: (B, n) 每instance异常分数
    y_bag: (B,) 弱标签
    pool: softmax / mean / topk (消融用)."""
    if pool == "softmax":
        w = torch.softmax(instance_scores, dim=-1)
        bag_score = (w * instance_scores).sum(dim=-1)
    elif pool == "mean":
        bag_score = instance_scores.mean(dim=-1)
    elif pool == "topk":
        k = max(1, instance_scores.shape[1] // 4)
        bag_score = instance_scores.topk(k, dim=-1).values.mean(dim=-1)
    else:
        raise ValueError(pool)
    # 将分数转 logit (分数可正可负, 用 sigmoid 包一层)
    return F.binary_cross_entropy_with_logits(bag_score, y_bag.float())


def compact_loss(z, centers, radii, soft=True, tau_c=1.0):
    """正常 instance 到所属球紧致 (第一篇权重0.20).
    z: (N, d).
    soft=True: 软加权(可微, 所有球都有梯度, 审稿友好), tau_c 控制锐度.
    soft=False: 硬分配(仅最近球有梯度, 导师C2原版)."""
    d2 = torch.cdist(z, centers).pow(2)  # (N,k) 对 z 可微
    R2 = radii.pow(2)                     # (k,)
    if soft:
        # 软归属权重 (越近权重越大)
        w = torch.softmax(-d2 / tau_c, dim=-1)  # (N,k)
        # 加权紧致距离
        compact_term = (w * d2).sum(dim=-1).mean()
    else:
        assign = d2.argmin(dim=-1)
        compact_term = d2.gather(1, assign.view(-1, 1)).squeeze(1).mean()
    # 半径惩罚: 防半径无限大
    return compact_term + R2.mean()


def balance_loss(z, centers, tau_b=1.0):
    """球均衡正则: 惩罚球间样本数极度不平衡 (防坍缩).
    各球软归属比例与均匀分布的 KL 散度."""
    d2 = torch.cdist(z, centers).pow(2)        # (N,k)
    w = torch.softmax(-d2 / tau_b, dim=-1)     # (N,k) 软归属
    p = w.mean(dim=0)                            # (k,) 各球平均归属比例
    uniform = torch.full_like(p, 1.0 / centers.shape[0])
    # KL(uniform || p): 当 p 偏离均匀(某球独大)时增大
    kl = (uniform * (torch.log(uniform + 1e-8) - torch.log(p + 1e-8))).sum()
    return kl


def separation_loss(centers, margin=1.5):
    """球中心间分离 (第一篇权重0.02, margin1.5)."""
    cdist = torch.cdist(centers, centers)
    mask = torch.triu(torch.ones_like(cdist, dtype=torch.bool), diagonal=1)
    pairs = cdist[mask]
    # hinge: max(0, margin - dist); 无配对时返回 0 (保持梯度连接到 centers)
    if pairs.numel() > 0:
        return F.relu(margin - pairs).mean()
    return (centers * 0.0).sum()


def margin_ranking_loss(scores_normal, scores_anomal, margin=1.0):
    """margin-ranking: 正常 bag 分数 < 异常 bag 分数 - margin (第一篇).
    仅当 batch 内两类 bag 同时出现."""
    if scores_normal.numel() == 0 or scores_anomal.numel() == 0:
        return torch.tensor(0.0, device=scores_normal.device)
    # 正常应低分, 异常应高分
    # 随机配对
    n = min(scores_normal.numel(), scores_anomal.numel())
    sn = scores_normal[torch.randperm(scores_normal.numel())[:n]]
    sa = scores_anomal[torch.randperm(scores_anomal.numel())[:n]]
    target = torch.ones_like(sa)
    return F.margin_ranking_loss(sa, sn, target, margin=margin)


def elbo_mog_prior(centers, mu0=None, sigma0=1.0):
    """混合高斯先验正则 (导师C2 公式迁移, 已逐符号核对 2026-08-02).
    导师原文公式(3) KL项: KL(N(μ,σ²I)‖N(0,I))
        = 0.5 * Σ((μ_d')² + (σ_d')² - log((σ_d')²) - 1)
    本篇对球中心施加同型约束 (中心作为隐空间均值).
    完整ELBO-MoG还含重构项, 此处先验正则聚焦KL(论文III-C)."""
    k, d = centers.shape
    if mu0 is None:
        mu0 = torch.zeros(d, device=centers.device)
    diff = centers - mu0.unsqueeze(0)  # (k,d) = μ_{d'}
    # 导师公式(3) KL项 (σ 设为常数, 仅约束 μ=centers)
    # KL = 0.5 * Σ((μ_d')² + σ² - log σ² - 1), σ 取 sigma0
    sigma2 = sigma0 ** 2
    kl_per_dim = 0.5 * (diff.pow(2) + sigma2 - torch.log(torch.tensor(sigma2)) - 1.0)
    return kl_per_dim.mean()


class CMHLoss(nn.Module):
    def __init__(self, w_bag=1.0, w_compact=0.20, w_sep=0.02, sep_margin=1.5,
                 w_rank=1.0, rank_margin=1.0, w_elbo=0.10, w_balance=0.10, pool="softmax"):
        super().__init__()
        self.w = dict(bag=w_bag, compact=w_compact, sep=w_sep, rank=w_rank,
                      elbo=w_elbo, balance=w_balance)
        self.sep_margin = sep_margin
        self.rank_margin = rank_margin
        self.pool = pool

    def forward(self, instance_scores, y_bag, z, centers, radii):
        """instance_scores: (B,n), y_bag:(B,), z:(B*n,d), centers/radii: 多球."""
        l_bag = bag_bce_loss(instance_scores, y_bag, self.pool)

        l_compact = compact_loss(z, centers, radii)
        l_sep = separation_loss(centers, self.sep_margin)
        l_elbo = elbo_mog_prior(centers)
        l_balance = balance_loss(z, centers)

        # ranking: 用 bag 内分数聚合, 按标签分组
        with torch.no_grad():
            if self.pool == "softmax":
                w = torch.softmax(instance_scores, dim=-1)
                bag_s = (w * instance_scores).sum(dim=-1)
            else:
                bag_s = instance_scores.mean(dim=-1)
        mask_n = (y_bag == 0)
        mask_a = (y_bag == 1)
        l_rank = margin_ranking_loss(bag_s[mask_n], bag_s[mask_a], self.rank_margin)

        total = (self.w["bag"] * l_bag + self.w["compact"] * l_compact
                 + self.w["sep"] * l_sep + self.w["rank"] * l_rank
                 + self.w["elbo"] * l_elbo + self.w["balance"] * l_balance)
        parts = {
            "total": total.item(), "bag": l_bag.item(),
            "compact": l_compact.item(), "sep": l_sep.item(),
            "rank": l_rank.item(), "elbo": l_elbo.item(),
            "balance": l_balance.item(),
        }
        return total, parts
