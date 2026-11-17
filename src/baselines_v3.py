"""W3 基线层 v3 (2026-11-17) — 全部走 W1 统一协议, 所有数字同源
==========================================================
总管令⑩: 官方实现替换/诚实改名. 本文件策略:
  PCA / IF / AE          — 经典方法, instance级特征 + 点级评测 (无需"官方", 定义即实现)
  DeepSVDD (faithful)    — Ruff et al. ICML'18 目标: 单中心紧致 min||z-c||²+λ‖W‖²,
                           中心为初始嵌入均值且 detach (原文做法), 无半径参数
  USAD (faithful)        — Audibert et al. KDD'20 目标: 双AE对抗, focus/generate 相位,
                           λ1,λ2 随 epoch 线性增长 (原文 Eq.5-7), 含重建-生成-对抗三损失
  DMSVDD                 — 改名 "multi-sphere SVDD-style (our impl. of Xing et al. 2024
                           objective)": 多球紧致+分离, 无解纠缠VAE (差异如实声明, 不冒名)
统一管线: four_block_split → Scaler(train块) → make_bags_v2 → instance分数 →
          point_scores(max聚合) → calib阈值 → evaluate_point (点级)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from data.protocol import (four_block_split, Scaler, make_bags_v2,
                           point_scores, threshold_from_calib, evaluate_point)

L, l, BAG_STRIDE = 128, 16, 64


def _pipeline(X, Y):
    """W1 统一管线: 返回各块 instance 数组与点级评测闭包."""
    sp = four_block_split(len(Y))
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)
    blocks = {}
    for name in ("train", "calib", "test"):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        blocks[name] = (insts, y_bag, starts + s, rel)

    def finish(inst_score_fn):
        s_all = np.full(len(Y), np.nan, dtype=np.float32)
        for name in ("calib", "test"):
            insts, yb, starts, rel = blocks[name]
            N, n = insts.shape[0], insts.shape[1]
            sc = np.concatenate([inst_score_fn(insts[i:i + 256]).reshape(-1)
                                 for i in range(0, N, 256)]).reshape(N, n)
            s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
            s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]
        thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                                   Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
        cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
        m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                           s_all[sp["test"][0]:sp["test"][1]], thr, cov)
        m["threshold"] = thr
        return m

    return blocks, finish


def _flatten(insts):
    return insts.reshape(insts.shape[0] * insts.shape[1], -1)  # (N*n, l*D)


# ---------- 经典基线 (instance 级, 点级评测) ----------
def baseline_pca(X, Y, seed=7, **kw):
    from sklearn.decomposition import PCA
    np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    pca = PCA(n_components=0.9, svd_solver="full").fit(_flatten(blocks["train"][0]))

    def score(insts):
        F = insts.reshape(insts.shape[0] * insts.shape[1], -1)
        recon = pca.inverse_transform(pca.transform(F))
        return ((F - recon) ** 2).mean(axis=1)
    return finish(score)


def baseline_if(X, Y, seed=7, **kw):
    from sklearn.ensemble import IsolationForest
    blocks, finish = _pipeline(X, Y)
    iso = IsolationForest(n_estimators=100, random_state=seed,
                          contamination="auto").fit(_flatten(blocks["train"][0]))

    def score(insts):
        return -iso.score_samples(insts.reshape(-1, insts.shape[-1] * l))
    return finish(score)


def baseline_ae(X, Y, seed=7, epochs=30, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    d = _flatten(blocks["train"][0]).shape[1]
    W = torch.from_numpy(_flatten(blocks["train"][0]))

    class AE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16))
            s.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
        def forward(s, x): return s.dec(s.enc(x))
    model = AE(); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 256):
            xb = W[perm[i:i + 256]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    def score(insts):
        with torch.no_grad():
            F = torch.from_numpy(insts.reshape(-1, insts.shape[-1] * l))
            return ((model(F) - F) ** 2).mean(1).numpy()
    return finish(score)


# ---------- DeepSVDD (faithful, Ruff et al. ICML 2018) ----------
def baseline_deep_svdd(X, Y, seed=7, epochs=30, **kw):
    """目标: min (1/N)Σ‖φ(x;θ) − c‖² + λ‖θ‖². c=初始嵌入均值(detach, 原文 §3.2).
    分数 = ‖φ(x)−c‖² (越大越异常). 无半径参数 (原文无)."""
    torch.manual_seed(seed); np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    d = _flatten(blocks["train"][0]).shape[1]
    W = torch.from_numpy(_flatten(blocks["train"][0]))
    enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16),
                        nn.ReLU(), nn.Linear(16, 8))
    with torch.no_grad():
        c = enc(W[torch.randperm(len(W))[:512]]).mean(0).detach()  # 原文: 中心=初始均值
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)  # λ‖θ‖² via wd
    for ep in range(epochs):
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 256):
            xb = W[perm[i:i + 256]]
            loss = ((enc(xb) - c) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()

    def score(insts):
        with torch.no_grad():
            F = torch.from_numpy(insts.reshape(-1, insts.shape[-1] * l))
            return ((enc(F) - c) ** 2).sum(1).numpy()
    return finish(score)


# ---------- USAD (faithful, Audibert et al. KDD 2020) ----------
def baseline_usad(X, Y, seed=7, epochs=30, **kw):
    """原文目标 (Eq.5-7, 含对抗相位与 λ 递增):
      AE1 = E1∘D1, AE2 = E2∘D2
      focus AE1 (k轮): L1 = ||X−D1(E1(X))||² + λ1·||X−D1(E2(X))||²
      focus AE2     : L2 = ||X−D2(E2(X))||² + λ2·||X−D2(E1(X))||²
      λi(k) = k/epochs (epoch 内线性增长)
    分数 = mean(|X−D2(E1(X))|, |X−D1(E2(X))|) (原文 reconstruction focus)."""
    torch.manual_seed(seed); np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    d = _flatten(blocks["train"][0]).shape[1]
    W = torch.from_numpy(_flatten(blocks["train"][0]))

    class Encoder(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16))
        def forward(s, x): return s.net(x)
    class Decoder(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
        def forward(s, z): return s.net(z)
    E1, D1, E2, D2 = Encoder(), Decoder(), Encoder(), Decoder()
    opt = torch.optim.Adam(list(E1.parameters()) + list(D1.parameters())
                           + list(E2.parameters()) + list(D2.parameters()), lr=1e-3)
    for ep in range(epochs):
        lam1, lam2 = (ep + 1) / epochs, (ep + 1) / epochs  # λ 线性增长 (原文)
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 256):
            xb = W[perm[i:i + 256]]
            # focus AE1 (对抗: E2 的编码经 D1 重建)
            x1 = D1(E1(xb)); x1g = D1(E2(xb))
            L1 = ((x1 - xb) ** 2).mean() + lam1 * ((x1g - xb) ** 2).mean()
            # focus AE2
            x2 = D2(E2(xb)); x2g = D2(E1(xb))
            L2 = ((x2 - xb) ** 2).mean() + lam2 * ((x2g - xb) ** 2).mean()
            loss = L1 + L2
            opt.zero_grad(); loss.backward(); opt.step()
    for m in (E1, D1, E2, D2):
        m.eval()

    def score(insts):
        with torch.no_grad():
            F = torch.from_numpy(insts.reshape(-1, insts.shape[-1] * l))
            r1 = D2(E1(F)); r2 = D1(E2(F))
            return (((r1 - F).abs() + (r2 - F).abs()) / 2).mean(1).numpy()
    return finish(score)


# ---------- 多球SVDD风格 (our implementation, 诚实改名) ----------
def baseline_ms_svdd(X, Y, seed=7, epochs=30, k=3, margin=1.5, **kw):
    """multi-sphere SVDD-style baseline — our implementation of the objective of
    Xing et al. (PR 2024): 多球紧致 + 球心分离. 不含其解纠缠VAE组件 (差异如实声明,
    论文中不得称 DMSVDD, 称 "MS-SVDD-style")."""
    torch.manual_seed(seed); np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    d = _flatten(blocks["train"][0]).shape[1]
    W = torch.from_numpy(_flatten(blocks["train"][0]))
    enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16))
    with torch.no_grad():
        from sklearn.cluster import KMeans
        z0 = enc(W[torch.randperm(len(W))[:512]]).numpy()
        centers = torch.from_numpy(
            KMeans(n_clusters=k, n_init=5, random_state=0).fit(z0).cluster_centers_).float()
    C = nn.Parameter(centers.clone())
    opt = torch.optim.Adam(list(enc.parameters()) + [C], lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 256):
            xb = W[perm[i:i + 256]]
            z = enc(xb)
            d2 = torch.cdist(z, C).pow(2)
            compact = d2.min(dim=-1).values.mean()          # 最近球紧致
            pd = torch.cdist(C, C)
            off = pd[~torch.eye(len(C), dtype=torch.bool)]
            sep = torch.relu(margin - off).mean()            # 球心分离
            loss = compact + 0.1 * sep
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()

    def score(insts):
        with torch.no_grad():
            F = torch.from_numpy(insts.reshape(-1, insts.shape[-1] * l))
            d2 = torch.cdist(enc(F), C).pow(2)
            return d2.min(dim=-1).values.numpy()
    return finish(score)


BASELINES_V3 = {
    "PCA": baseline_pca,
    "IF": baseline_if,
    "AE": baseline_ae,
    "DeepSVDD": baseline_deep_svdd,
    "USAD": baseline_usad,
    "MS-SVDD-style": baseline_ms_svdd,
}

if __name__ == "__main__":
    import json, time
    from run_v3 import load
    ds = sys.argv[1] if len(sys.argv) > 1 else "SMD"
    X, Y = load(ds)
    out = {}
    for name, fn in BASELINES_V3.items():
        t0 = time.time()
        try:
            m = fn(X, Y, seed=7)
            out[name] = m
            print(f"{name:14s} AUPRC={m['AUPRC']:.4f} PF1={m['PointF1']:.4f} "
                  f"EF1={m['EventF1_one2one']:.4f} FAR={m['FAR_point']:.4f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"{name:14s} FAIL: {e}")
            out[name] = {"error": str(e)[:200]}
    p = os.path.join(os.path.dirname(__file__), "..", "results", f"baselines_v3_{ds}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print("saved:", p)
