"""CMH-MIL v3 损失 (W2 重建版, 2026-11-17)
==========================================
相对 v2 的修改 (总管令 W2, 均有裁定依据):

[⑥ Eq(4)对齐 — 总管裁定: 代码实现论文式]
  v2: 对每个正样本 p_i 计算 (-p_i + logsumexp(all)) 再取均值  ← 逐正样本均值
  v3: 每个 anchor 计算 logsumexp(all) - logsumexp(pos)          ← 论文式
      论文 Eq(4): -log[ Σ_{m+} exp(s+) / Σ_{m'} exp(s') ]
      数值稳定实现: logsumexp(all) - logsumexp(pos)
  差异本质: 论文式分子是正样本指数和(联合), v2是逐正样本均值的log概率.

[⑨ dispersion 正则 — 总管裁定案A: 诚实命名]
  v2: return -off_diag(pdist).mean()  注释称"均匀性"  ← 注释与行为不符(R1)
  v3: return pdist.mean() = mean(1 - cos)
      命名: mean-cosine dispersion (最小化平均余弦相似度 → 分散)
      与 v2 数学等价 (差常数 +1, 优化方向相同: C↓ → 分散)
      明确声明: 非 Wang & Isola uniformity — 只约束表征均值 (‖Σz‖),
      不约束分布形状 (反向双簇与均匀分布损失相同, GPT‖Σu‖²论证)

[⑦ bag BCE 解耦]
  v2: CMHLossV2.forward 内部用 softmax 重新聚合 bag 分数 (与模型 pool 不一致)
  v3: bag BCE 直接消费模型 forward 输出的 bag_score (单一聚合来源)

[⑤ 半径无下界] 本版未动 (两案待总管批准): v3.radius_quantile 开关预留
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dispersion_loss(bag_embeddings):
    """⑨ mean-cosine dispersion 正则 (P0-1 修复版, 2026-11-18 三外审抓出方向反转).
    正确目标: L = mean_{m≠m'} cos(z_m, z_m')  →  最小化平均余弦相似度 → 表征分散.
    求导验证: L=C, dL/dC=+1, 最小化→C↓→分散 ✓
    历史(错误记录, 供教训档案):
      v2 代码  -mean(1-cos)=C-1  → 分散 ✓ (行为对)
      v3 初版  +mean(1-cos)=1-C  → 聚集 ✗ (照抄案A字面公式未求导, 违反纪律一)
      案A裁定原文 "L=mean(1-sim)" 字面即 1-C=聚集, 与裁定意图(分散)相反——
      裁定书公式笔误 + 实施未复验, 双重失效. 本版以行为意图(分散)为准.
    注: 非 W&I 分布均匀性 (仅约束表征均值, 不约束分布形状)."""
    z = F.normalize(bag_embeddings, dim=-1)
    cos = z @ z.t()
    off_diag = cos[~torch.eye(len(z), dtype=torch.bool, device=z.device)]
    return off_diag.mean()


def infonce_loss(bag_embeddings, y_bag, temperature=0.1):
    """⑥ bag 级 InfoNCE (论文 Eq(4) 逐符号实现, 数值稳定).
    anchor = 每个 normal bag; 正样本 = 其他 normal bag; 负样本 = anomaly bag.
    L_m = -log[ Σ_{+} exp(s+/τ) / Σ_{all} exp(s/τ) ] = logsumexp(all) - logsumexp(pos)
    """
    z = F.normalize(bag_embeddings, dim=-1)
    sim = (z @ z.t()) / temperature
    is_normal = (y_bag == 0)
    is_anomaly = (y_bag == 1)
    if is_normal.sum() < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    if is_anomaly.sum() == 0:
        return dispersion_loss(bag_embeddings)  # 全正常分支 → dispersion
    normal_idx = torch.where(is_normal)[0]
    anomaly_idx = torch.where(is_anomaly)[0]
    n_n = len(normal_idx)
    loss = 0.0
    n_valid = 0
    for ii in range(n_n):
        i = normal_idx[ii]
        self_mask = torch.zeros(n_n, dtype=torch.bool, device=z.device)
        self_mask[ii] = True
        pos = sim[i, normal_idx][~self_mask]      # 其他 normal bag (排除 anchor 自己)
        if len(pos) == 0:
            continue
        neg = sim[i, anomaly_idx]
        all_ = torch.cat([pos, neg])
        loss = loss + (torch.logsumexp(all_, dim=0) - torch.logsumexp(pos, dim=0))
        n_valid += 1
    if n_valid == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return loss / n_valid


def topk_rank_loss(instance_scores, y_bag, k=None, margin=1.0):
    """TopK 排序损失 (沿用 v2 语义, 未改动 — ⑤批准后若改分数定义再联动)."""
    B, n = instance_scores.shape
    if k is None:
        k = max(1, n // 4)
    pos_mask = (y_bag == 1)
    neg_mask = (y_bag == 0)
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        pos_topk = instance_scores[pos_mask].topk(min(k, n), dim=-1).values.mean(dim=-1).mean()
        neg_topk = instance_scores[neg_mask].topk(min(k, n), dim=-1).values.mean(dim=-1).mean()
        return F.relu(margin + neg_topk - pos_topk)
    elif neg_mask.sum() > 0:
        neg_topk = instance_scores[neg_mask].topk(min(k, n), dim=-1).values
        return neg_topk.mean() + 0.1 * neg_topk.var()
    return torch.tensor(0.0, device=instance_scores.device, requires_grad=True)


def compact_loss(z, centers, radii, soft=True, tau_c=1.0, w_radius=0.1,
                 radius_mode="quantile"):
    """紧致损失. ⑤b2: quantile 模式下 radius_term 删除 (半径为统计量,
    无梯度通道, 惩罚无意义). learnable 模式保留 w_radius 旋钮(历史复现)."""
    d2 = torch.cdist(z, centers).pow(2)
    if soft:
        w = torch.softmax(-d2 / tau_c, dim=-1)
        compact_term = (w * d2).sum(dim=-1).mean()
    else:
        compact_term = d2.min(dim=-1).values.mean()
    if radius_mode == "learnable":
        return compact_term + w_radius * radii.pow(2).mean()
    return compact_term


def sep_loss(centers, margin=1.5):
    pdist = torch.cdist(centers, centers)
    off_diag = pdist[~torch.eye(len(centers), dtype=torch.bool, device=centers.device)]
    return F.relu(margin - off_diag).mean()


def balance_loss(z, centers, k):
    d2 = torch.cdist(z, centers).pow(2)
    soft_assign = torch.softmax(-d2, dim=-1)
    prop = soft_assign.mean(dim=0)
    target = torch.ones_like(prop) / k
    return F.kl_div((prop + 1e-8).log(), target, reduction='sum')


class CMHLossV3(nn.Module):
    """v3 总损失. bag BCE 消费模型 bag_score (⑦解耦)."""

    def __init__(self, w_bag=1.0, w_compact=0.10, w_sep=0.05, w_topk=1.0,
                 w_contrast=0.5, w_balance=0.10, topk_k=None, topk_margin=1.0,
                 contrast_tau=0.1, sep_margin=1.5, w_radius=0.1,
                 radius_mode="quantile"):
        super().__init__()
        self.w_bag, self.w_compact, self.w_sep = w_bag, w_compact, w_sep
        self.w_topk, self.w_contrast, self.w_balance = w_topk, w_contrast, w_balance
        self.topk_k, self.topk_margin = topk_k, topk_margin
        self.contrast_tau, self.sep_margin = contrast_tau, sep_margin
        self.w_radius = w_radius
        self.radius_mode = radius_mode

    def forward(self, bag_score, y_bag, instance_scores, z, bag_z, centers, radii):
        """bag_score: (B,) 模型聚合后的 bag 分数 (⑦ 单一聚合来源)."""
        l_bag = F.binary_cross_entropy_with_logits(bag_score, y_bag.float())
        l_topk = topk_rank_loss(instance_scores, y_bag, k=self.topk_k, margin=self.topk_margin)
        l_contrast = infonce_loss(bag_z, y_bag, temperature=self.contrast_tau)
        l_compact = compact_loss(z, centers, radii, w_radius=self.w_radius,
                                 radius_mode=self.radius_mode)
        l_sep = sep_loss(centers, margin=self.sep_margin)
        l_balance = balance_loss(z, centers, centers.shape[0])
        total = (self.w_bag * l_bag + self.w_topk * l_topk + self.w_contrast * l_contrast
                 + self.w_compact * l_compact + self.w_sep * l_sep + self.w_balance * l_balance)
        return total, {
            'bag': float(l_bag), 'topk': float(l_topk), 'contrast': float(l_contrast),
            'compact': float(l_compact), 'sep': float(l_sep), 'balance': float(l_balance),
        }
