"""CMH-MIL v2 损失：对比学习(InfoNCE) + TopK排序 + bag BCE
核心创新：首个将对比学习引入弱监督 MIL 时序异常检测。
= bag_BCE + compact + sep + TopK排序 + InfoNCE对比 + balance
去掉无效的 ELBO-MoG (v1消融已证伪)。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bag_bce_loss(instance_scores, y_bag, pool="softmax"):
    """bag 分数聚合 -> bag BCE (保留 v1 基础)."""
    if pool == "softmax":
        w = torch.softmax(instance_scores, dim=-1)
        bag_score = (w * instance_scores).sum(dim=-1)
    elif pool == "topk":
        k = max(1, instance_scores.shape[1] // 4)
        bag_score = instance_scores.topk(k, dim=-1).values.mean(dim=-1)
    else:
        bag_score = instance_scores.mean(dim=-1)
    return F.binary_cross_entropy_with_logits(bag_score, y_bag.float())


def topk_rank_loss(instance_scores, y_bag, k=None, margin=1.0):
    """TopK MIL 排序损失 (创新点2).
    两类都有时: 异常bag TopK要大, 正常bag TopK要小 (排序间隔)
    只有正常bag时(标准MTSAD): 正常bag的TopK分数尽量小 (单类紧致),
        同时用分数方差作为对比信号 (让模型关注最高分的难例).
    instance_scores: (B, n)
    y_bag: (B,) 0=正常 1=异常
    """
    B, n = instance_scores.shape
    if k is None:
        k = max(1, n // 4)
    pos_mask = (y_bag == 1)
    neg_mask = (y_bag == 0)
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        # 两类都有: 标准排序间隔
        pos_scores = instance_scores[pos_mask]
        pos_topk = pos_scores.topk(min(k, n), dim=-1).values.mean(dim=-1).mean()
        neg_scores = instance_scores[neg_mask]
        neg_topk = neg_scores.topk(min(k, n), dim=-1).values.mean(dim=-1).mean()
        return F.relu(margin + neg_topk - pos_topk)
    elif neg_mask.sum() > 0:
        # 只有正常 bag (标准MTSAD): TopK紧致 + 方差正则
        # 正常 bag 的 TopK 分数应尽量小 (紧致)
        neg_scores = instance_scores[neg_mask]
        neg_topk = neg_scores.topk(min(k, n), dim=-1).values
        # TopK 均值 (要小)
        topk_mean = neg_topk.mean()
        # TopK 方差 (要小, 让难例也集中, 不发散)
        topk_var = neg_topk.var()
        return topk_mean + 0.1 * topk_var
    else:
        return torch.tensor(0.0, device=instance_scores.device, requires_grad=True)


def contrastive_loss(bag_embeddings, y_bag, temperature=0.1):
    """Bag 级对比学习损失 (创新点1, 核心).
    正常bag互为正样本对, 异常bag作为负样本.
    bag_embeddings: (B, d) 每个 bag 的池化表征
    y_bag: (B,) 0=正常 1=异常
    """
    B, d = bag_embeddings.shape
    device = bag_embeddings.device
    # L2 归一化 (余弦相似度)
    z = F.normalize(bag_embeddings, dim=-1)
    # 相似度矩阵 (B, B)
    sim = z @ z.t() / temperature
    # 正样本掩码: 同为正常 bag (y=0)
    # 负样本: 异常 bag (y=1)
    is_normal = (y_bag == 0)
    is_anomaly = (y_bag == 1)
    # 只在正常 bag 之间做对比 (正常互为正, 异常为负)
    if is_normal.sum() < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    # 标准MTSAD: 全正常batch时, 让正常bag表征聚集 (均匀性正则)
    if is_anomaly.sum() == 0:
        # 所有 bag 互为正样本, 最小化两两距离 (聚集)
        all_z = z[is_normal]
        # 均匀性: 最小化 bag 间余弦距离 (让表征紧凑且均匀)
        pdist = 1 - (all_z @ all_z.t())  # 余弦距离
        off_diag = pdist[~torch.eye(len(all_z), dtype=torch.bool, device=device)]
        return -off_diag.mean()  # 负号 = 最小化距离 = 最大化相似度
    # 有正有负: 标准 InfoNCE
    loss = 0.0
    n_valid = 0
    normal_idx = torch.where(is_normal)[0]
    anomaly_idx = torch.where(is_anomaly)[0]
    for i in normal_idx:
        # 正样本相似度 (与其他正常 bag)
        pos_sim = sim[i, normal_idx]  # 含自己, 对角线=1
        # 排除自己 (对角线)
        mask_self = torch.zeros_like(pos_sim, dtype=torch.bool)
        mask_self[normal_idx == i] = True
        pos_sim = pos_sim[~mask_self]
        if len(pos_sim) == 0:
            continue
        # 负样本相似度 (与异常 bag)
        neg_sim = sim[i, anomaly_idx]
        # InfoNCE: log( exp(pos) / (exp(pos) + sum exp(neg)) )
        # 取所有正样本的平均
        all_sim = torch.cat([pos_sim, neg_sim])  # 先排正, 后排负
        # 分母 = 所有 (正+负)
        log_sum_exp = torch.logsumexp(all_sim, dim=0)
        # 对每个正样本计算 -pos + logsumexp(all)
        loss += (-pos_sim + log_sum_exp).mean()
        n_valid += 1
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss / n_valid


def compact_loss(z, centers, radii, soft=True, tau_c=1.0):
    """正常 instance 到所属球紧致 (保留 v1)."""
    d2 = torch.cdist(z, centers).pow(2)
    if soft:
        w = torch.softmax(-d2 / tau_c, dim=-1)
        compact_term = (w * d2).sum(dim=-1).mean()
    else:
        compact_term = d2.min(dim=-1).values.mean()
    radius_term = radii.pow(2).mean()
    return compact_term + 0.1 * radius_term


def sep_loss(centers, margin=1.5):
    """球心分离 (保留 v1)."""
    pdist = torch.cdist(centers, centers)
    off_diag = pdist[~torch.eye(len(centers), dtype=torch.bool, device=centers.device)]
    return F.relu(margin - off_diag).mean()


def balance_loss(z, centers, k):
    """KL 平衡正则 (保留 v1, 防球坍缩)."""
    d2 = torch.cdist(z, centers).pow(2)
    soft_assign = torch.softmax(-d2, dim=-1)  # (N, k)
    prop = soft_assign.mean(dim=0)  # (k,)
    target = torch.ones_like(prop) / k
    return F.kl_div((prop + 1e-8).log(), target, reduction='sum')


class CMHLossV2(nn.Module):
    """CMH-MIL v2 总损失."""
    def __init__(self, w_bag=1.0, w_compact=0.10, w_sep=0.05, w_topk=1.0,
                 w_contrast=0.5, w_balance=0.10, topk_k=None, topk_margin=1.0,
                 contrast_tau=0.1, sep_margin=1.5):
        super().__init__()
        self.w_bag = w_bag
        self.w_compact = w_compact
        self.w_sep = w_sep
        self.w_topk = w_topk
        self.w_contrast = w_contrast
        self.w_balance = w_balance
        self.topk_k = topk_k
        self.topk_margin = topk_margin
        self.contrast_tau = contrast_tau
        self.sep_margin = sep_margin

    def forward(self, instance_scores, y_bag, z, bag_z, centers, radii):
        """
        instance_scores: (B, n)
        y_bag: (B,)
        z: (B*n, d) instance 表征
        bag_z: (B, d) bag 池化表征 (用于对比学习)
        centers: (k, d), radii: (k,)
        """
        l_bag = bag_bce_loss(instance_scores, y_bag, pool="softmax")
        l_topk = topk_rank_loss(instance_scores, y_bag, k=self.topk_k, margin=self.topk_margin)
        l_contrast = contrastive_loss(bag_z, y_bag, temperature=self.contrast_tau)
        l_compact = compact_loss(z, centers, radii)
        l_sep = sep_loss(centers, margin=self.sep_margin)
        k = centers.shape[0]
        l_balance = balance_loss(z, centers, k)
        total = (self.w_bag * l_bag + self.w_topk * l_topk + self.w_contrast * l_contrast
                 + self.w_compact * l_compact + self.w_sep * l_sep + self.w_balance * l_balance)
        # 返回总损失 + 各分量(用于日志)
        return total, {
            'bag': float(l_bag.item()) if hasattr(l_bag, 'item') else 0,
            'topk': float(l_topk.item()) if hasattr(l_topk, 'item') else 0,
            'contrast': float(l_contrast.item()) if hasattr(l_contrast, 'item') else 0,
            'compact': float(l_compact.item()),
            'sep': float(l_sep.item()),
            'balance': float(l_balance.item()),
        }
