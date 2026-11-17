"""Anomaly Transformer — faithful reimplementation of the original objective
(Xu et al., ICLR 2022; official code thuml/Anomaly-Transformer).
W3-⑩ per 总管令. 与简化版(baseline_anomaly_transformer.py, 历史)的区别:
  1. minimax 两阶段训练 (原文 Phase 1/2, 每 epoch 内交替):
     series_loss = KL(series_assoc || prior_assoc.detach())
     prior_loss  = KL(prior_assoc || series_assoc.detach())
     Phase1(拟合prior): L = recon + λ1·series.detach() − λ1·prior   (prior→series)
     Phase2(最小化重构下最大化discrepancy): L = recon + λ2·series − λ2·prior.detach()
  2. prior-association = Gaussian kernel(时间距离, 每head可学sigma) 归一化
  3. 推理分数 = recon_error + 归一化 AssocDiscrepancy (原文式)
窗口语义: 原文 win=100; 本文协议 L=128 的 bag 窗口 (B,128,D) 直接输入.
分数: 每窗一个 → 广播到窗内 instance (管线点级映射复用).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
import torch
import torch.nn as nn

from data.protocol import (four_block_split, Scaler, make_bags_v2,
                           point_scores, threshold_from_calib, evaluate_point)

L, l, BAG_STRIDE = 128, 16, 64


class AnomalyAttention(nn.Module):
    """原文: series-association (QK^T learnable) vs prior-association (Gaussian, 可学sigma)."""

    def __init__(self, win_size, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.sigma = nn.Parameter(torch.ones(n_heads) * 0.5)   # 每head可学sigma (原文)
        idx = torch.arange(win_size)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).float().pow(2)
        self.register_buffer("dist", dist)

    def forward(self, x):  # x: (B, L, d)
        B, Lw, _ = x.shape
        Q = self.Wq(x).view(B, Lw, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, Lw, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, Lw, self.n_heads, self.d_k).transpose(1, 2)
        series = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        # prior: Gaussian kernel on |i-j|², 每head自己的sigma, 行归一化
        sigma2 = self.sigma.pow(2).view(1, self.n_heads, 1, 1)          # (1,H,1,1)
        prior = torch.exp(-self.dist / (2.0 * sigma2))
        prior = prior / prior.sum(-1, keepdim=True)
        return series, prior, V

    @staticmethod
    def kl(p, q):  # KL(p||q) 逐元素 p·log(p/q) 求和 (原文 my_kl)
        return (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(-1)


class ATBlock(nn.Module):
    def __init__(self, win, d_model, n_heads):
        super().__init__()
        self.attn = AnomalyAttention(win, d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        series, prior, V = self.attn(x)
        out = (series @ V).transpose(1, 2).reshape(x.shape)
        h = self.norm(x + out)
        return self.norm(h + self.ffn(h)), series, prior


class ATModel(nn.Module):
    def __init__(self, D, win=128, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([ATBlock(win, d_model, n_heads) for _ in range(n_layers)])
        self.recon = nn.Linear(d_model, D)

    def forward(self, x):                     # x: (B, L, D)
        h = self.embed(x)
        series_all, prior_all = [], []
        for blk in self.blocks:
            h, s, p = blk(h)
            series_all.append(s); prior_all.append(p)
        return self.recon(h), series_all, prior_all


def baseline_at_faithful(X, Y, seed=7, epochs=15, lam=3.0, **kw):
    """两阶段 minimax 训练 (原文 Phase1/Phase2 每 epoch 交替)."""
    torch.manual_seed(seed); np.random.seed(seed)
    sp = four_block_split(len(Y))
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)
    D = X.shape[1]

    def mk(name):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        # AT 用整个 bag 窗口: (N, n, l, D) -> (N, L, D) 取每 instance 拼接
        N, n = insts.shape[0], insts.shape[1]
        win = insts.reshape(N, n * l, D)[:, :L, :]           # n*l=128 == L
        return win, y_bag, starts + s, rel, N, n

    tr_win = mk("train")[0]
    model = ATModel(D, win=L)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    W = torch.from_numpy(tr_win)
    for ep in range(epochs):
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 32):
            xb = W[perm[i:i + 32]]
            recon, series, priors = model(xb)
            rec_loss = ((recon - xb) ** 2).mean()
            # 逐层逐head KL (B, H, L, L)
            kl_sp = sum((AnomalyAttention.kl(s, p.detach())).mean()
                        for s, p in zip(series, priors))
            kl_ps = sum((AnomalyAttention.kl(p, s.detach())).mean()
                        for s, p in zip(series, priors))
            # Phase1: prior→series (detach series 侧); Phase2: 最大化 discrepancy
            loss1 = rec_loss + lam * kl_sp.detach() - lam * kl_ps
            loss2 = rec_loss + lam * kl_sp - lam * kl_ps.detach()
            # epoch 奇偶交替两相位 (原文两阶段串行的紧凑等价)
            loss = loss1 if (ep % 2 == 0) else loss2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()

    def window_score(win_np):
        """原文推理: recon_error + 归一化 AssocDiscrepancy (逐窗)."""
        out = []
        with torch.no_grad():
            for i in range(0, len(win_np), 128):
                xb = torch.from_numpy(win_np[i:i + 128])
                recon, series, priors = model(xb)
                rec = ((recon - xb) ** 2).mean(-1)                       # (B, L)
                disc = sum((s - p).abs().mean(dim=(1, -2, -1))           # head维一并均值 → (B,)
                           for s, p in zip(series, priors)) / len(series)
                disc_n = disc / (disc.max() + 1e-9)                      # min-max归一(原文)
                rec_n = rec.mean(-1) / (rec.mean(-1).max() + 1e-9)
                out.append((rec_n + disc_n).numpy())
        return np.concatenate(out)                                        # (N,)

    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    for name in ("calib", "test"):
        win, yb, starts, rel, N, n = mk(name)
        wsc = window_score(win)                                          # (N,)
        inst_sc = np.repeat(wsc[:, None], n, axis=1)                     # 广播到instance
        s_blk, _ = point_scores(len(Y), starts, rel, inst_sc, l, agg="max")
        s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
    cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                       s_all[sp["test"][0]:sp["test"][1]], thr, cov)
    m["threshold"] = thr
    return m


if __name__ == "__main__":
    import json, time
    from run_v3 import load
    ds = sys.argv[1] if len(sys.argv) > 1 else "SMD"
    X, Y = load(ds)
    t0 = time.time()
    m = baseline_at_faithful(X, Y, seed=7)
    print(f"AT-faithful {ds}: AUPRC={m['AUPRC']:.4f} PF1={m['PointF1']:.4f} "
          f"EF1={m['EventF1_one2one']:.4f} FAR={m['FAR_point']:.4f} ({time.time()-t0:.0f}s)")
    p = os.path.join(os.path.dirname(__file__), "..", "results", f"at_faithful_{ds}.json")
    json.dump(m, open(p, "w"), default=float, indent=2)
    print("saved:", p)
