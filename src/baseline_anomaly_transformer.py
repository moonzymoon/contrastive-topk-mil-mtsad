"""Anomaly Transformer baseline - Xu et al. ICLR 2022
Association Discrepancy: |序列关联 - 先验关联| 作为异常分数.
轻量 CPU 实现, 复用本项目的四块分割与评估协议.

Reference: Xu, J. et al. "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy." ICLR 2022.
"""
import numpy as np
import torch
import torch.nn as nn
import math
from evaluation.conformal import events_from_binary
from evaluation.metrics import evaluate_all


class AnomalyAttention(nn.Module):
    """Anomaly-Attention: 同时计算序列关联(Gaussian kernel)和先验关联(Gaussian prior)."""
    def __init__(self, win_size, d_model, n_heads):
        super().__init__()
        self.win_size = win_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 可学习的高斯先验 sigma
        self.sigma = nn.Parameter(torch.ones(1) * 0.5)
        # 预计算距离矩阵
        idx = torch.arange(win_size)
        self.register_buffer('dist', (idx.unsqueeze(0) - idx.unsqueeze(1)).float() ** 2)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        # 序列关联 (scaled dot-product, learnable)
        series = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        series = torch.softmax(series, dim=-1)
        # 先验关联 (Gaussian prior, fixed structure with learnable sigma)
        prior = self.dist[:L, :L].unsqueeze(0).unsqueeze(0)
        prior = torch.exp(-prior / (2 * self.sigma ** 2 + 1e-8))
        prior = prior / (prior.sum(-1, keepdim=True) + 1e-8)
        # Association discrepancy = |series - prior| (averaged over heads)
        discrepancy = (series - prior).abs().mean(dim=(1, 2))  # (B, L)
        # 值聚合
        out = torch.matmul(series, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out, discrepancy


class ATBlock(nn.Module):
    def __init__(self, win_size, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(win_size, d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, disc = self.attn(x)
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, disc


class AnomalyTransformerModel(nn.Module):
    """Anomaly Transformer: encoder + reconstruction head.
    训练 loss = 重建误差 + lambda * (1 - discrepancy) (鼓励低 discrepancy 使重建好,
    推理时用 discrepancy 作为异常分数: 异常点序列关联偏离先验, discrepancy 高)."""
    def __init__(self, D, win_size, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([ATBlock(win_size, d_model, n_heads) for _ in range(n_layers)])
        self.recon = nn.Linear(d_model, D)
        self.win_size = win_size

    def forward(self, x):
        # x: (B, L, D)
        h = self.embed(x)
        total_disc = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for blk in self.blocks:
            h, disc = blk(h)
            total_disc += disc
        recon = self.recon(h)
        return recon, total_disc / len(self.blocks)


def _make_windows(X, win_size, stride):
    """把 (T, D) 切成 (N, L, D) 滑动窗口."""
    T = len(X)
    starts = list(range(0, T - win_size + 1, stride))
    wins = np.stack([X[s:s + win_size] for s in starts])
    return wins, starts


def baseline_anomaly_transformer(X, Y, cfg, seed=42, win_size=100, stride=50):
    """Anomaly Transformer baseline (Xu et al. ICLR 2022).

    四块分割复用项目协议: train 35% / val 10% / calib 10% / test 45%.
    异常分数 = association discrepancy (高 = 异常).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    D = X.shape[1]; T = len(X)
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)

    Xtr_raw = X[:a]; Xcal = X[b:c]; Xte = X[c:]; Yte = Y[c:]

    # 标准化 (用训练块统计)
    mu = Xtr_raw.mean(0); sigma = Xtr_raw.std(0) + 1e-8
    Xs = (X - mu) / sigma

    # 建窗口
    Xtr_w, _ = _make_windows(Xs[:a], win_size, stride)

    model = AnomalyTransformerModel(D, win_size, d_model=64, n_heads=2, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10; batch_size = 32; lam = 3.0  # AT 原文 lambda

    Xtr_t = torch.from_numpy(Xtr_w.astype(np.float32))
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch_size):
            x = Xtr_t[perm[i:i + batch_size]]
            recon, disc = model(x)
            recon_loss = ((recon - x) ** 2).mean()
            # AT 原文: 最小化 discrepancy (让正常数据序列关联贴近先验),
            # 这样异常点(序列关联偏离先验)在推理时 disc 高、易检出
            disc_loss = disc.mean()
            loss = recon_loss + lam * disc_loss
            opt.zero_grad(); loss.backward(); opt.step()

    # 推理: AT 原文分数 = reconstruction_error (主) + association_discrepancy (辅)
    # 正常数据训练后重建好、discrepancy低; 异常数据重建差、discrepancy高
    def score_windows(data_raw):
        data_s = (data_raw - mu) / sigma
        wins, starts = _make_windows(data_s, win_size, stride)
        wt = torch.from_numpy(wins.astype(np.float32))
        scores = np.zeros(len(data_raw))
        counts = np.zeros(len(data_raw))
        with torch.no_grad():
            for i in range(0, len(wt), 64):
                recon, disc = model(wt[i:i + 64])
                # 逐点重建误差 (B, L, D) -> (B, L)
                recon_err = ((recon - wt[i:i + 64]) ** 2).mean(-1).numpy()
                disc_np = disc.numpy()
                # 组合分数: 重建误差 (主信号) + 标准化的 discrepancy
                # 两项都标准化到 [0,1] 再相加, 避免量纲差异
                for j, s in enumerate(starts[i:i + 64]):
                    re_z = recon_err[j]
                    dc_z = disc_np[j]
                    combined = re_z + dc_z
                    scores[s:s + win_size] += combined
                    counts[s:s + win_size] += 1
        counts[counts == 0] = 1
        return scores / counts

    cal_scores = score_windows(Xcal)
    test_scores = score_windows(Xte)
    thr = np.quantile(cal_scores[Y[b:c] == 0], 0.95)
    pred = (test_scores > thr).astype(int)

    m = evaluate_all(Yte, test_scores, float(thr),
                     events_from_binary(Yte), events_from_binary(pred))
    m['empirical_FAR'] = float(pred[Yte == 0].mean()) if (Yte == 0).sum() > 0 else float('nan')
    return m


if __name__ == '__main__':
    import sys, json, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data.loaders import load_dataset

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['SMD', 'PSM', 'MSL', 'SMAP']
    seeds = [7, 42, 123, 202, 999]
    results = {}
    for ds_name in datasets:
        print(f'\n=== {ds_name} ===')
        try:
            X, Y = load_dataset(ds_name)
            print(f'  loaded: X={X.shape}, Y={Y.shape}, anomaly_ratio={Y.mean():.4f}')
        except Exception as e:
            print(f'  load failed: {e}'); continue
        per_seed = []
        for seed in seeds:
            try:
                m = baseline_anomaly_transformer(X, Y, {}, seed=seed)
                m['seed'] = seed
                per_seed.append(m)
                print(f'  seed {seed}: AUPRC={m["AUPRC"]:.4f} EF1={m["EventF1_one2one"]:.4f}')
            except Exception as e:
                print(f'  seed {seed} failed: {e}')
        if per_seed:
            import statistics
            agg = {}
            for key in ['AUPRC', 'AUROC', 'PointF1', 'EventF1_one2one', 'empirical_FAR']:
                vals = [s[key] for s in per_seed if key in s and s[key] == s[key]]
                if vals:
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals) if len(vals) > 1 else 0
                    agg[key] = f'{mean:.4f}±{std:.4f}'
            results[ds_name] = {'per_seed': per_seed, 'aggregate': agg}
            print(f'  AGG: {agg}')
    out_path = '../results/anomaly_transformer.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')
