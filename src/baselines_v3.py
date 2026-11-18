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
    # bias=False 全层 (Ruff et al. 2018 §3.3 Prop.2: 有偏线性层存在常数映射 h≡c 坍缩解,
    # 原文明确去除; 三外审 P0-2 抓出此前遗漏)
    enc = nn.Sequential(nn.Linear(d, 64, bias=False), nn.ReLU(),
                        nn.Linear(64, 16, bias=False), nn.ReLU(),
                        nn.Linear(16, 8, bias=False))
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


# ---------- USAD (faithful v2, Audibert et al. KDD 2020, per official code) ----------
def baseline_usad(X, Y, seed=7, epochs=10, **kw):
    """忠实版 (三外审 A5 修复, 2026-11-18): 逐行对照官方 UsadModel/training/testing —
      结构: 单共享 encoder + decoder1 + decoder2 (官方 42-47 行)
      训练: 每 batch 双阶段, 各自独立 optimizer (官方 82-101 行):
        z=E(w); w1=D1(z); w2=D2(z); w3=D2(E(w1))          (官方 50-53 行)
        loss1 = (1/n)·MSE(w,w1) + (1-1/n)·MSE(w,w3)       ← AE1 阶段全正号
        loss2 = (1/n)·MSE(w,w2) − (1-1/n)·MSE(w,w3)       ← AE2 阶段负号对抗项 ✓
        n = epoch+1 (λ=1−1/n 随 epoch 递增); AE2 阶段前重新前向 (官方 98 行)
      分数: α·MSE(w,w1) + β·MSE(w,w2), α=β=0.5 (官方 testing 109-116 行)
      输入: 官方对窗口 minmax→sigmoid 解码; 此处 instance 展平后 minmax[0,1],
        decoder 末端 Sigmoid (保持官方数值域), 训练/推理一致."""
    torch.manual_seed(seed); np.random.seed(seed)
    blocks, finish = _pipeline(X, Y)
    d = _flatten(blocks["train"][0]).shape[1]

    def _mm(t):  # minmax -> [0,1] (逐行)
        mn, mx = t.min(1, keepdim=True).values, t.max(1, keepdim=True).values
        return (t - mn) / (mx - mn + 1e-8)

    W = _mm(torch.from_numpy(_flatten(blocks["train"][0])))

    class Encoder(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(d, max(d // 2, 16)); s.l2 = nn.Linear(max(d // 2, 16), max(d // 4, 8))
            s.l3 = nn.Linear(max(d // 4, 8), 10)
        def forward(s, x):
            x = torch.relu(s.l1(x)); x = torch.relu(s.l2(x))
            return torch.relu(s.l3(x))
    class Decoder(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(10, max(d // 4, 8)); s.l2 = nn.Linear(max(d // 4, 8), max(d // 2, 16))
            s.l3 = nn.Linear(max(d // 2, 16), d)
        def forward(s, z):
            z = torch.relu(s.l1(z)); z = torch.relu(s.l2(z))
            return torch.sigmoid(s.l3(z))
    E, D1, D2 = Encoder(), Decoder(), Decoder()
    opt1 = torch.optim.Adam(list(E.parameters()) + list(D1.parameters()), lr=1e-3)
    opt2 = torch.optim.Adam(list(E.parameters()) + list(D2.parameters()), lr=1e-3)

    def fwd(xb):
        z = E(xb); w1 = D1(z); w2 = D2(z); w3 = D2(E(w1))
        return w1, w2, w3

    for ep in range(epochs):
        n = ep + 1
        perm = torch.randperm(len(W))
        for i in range(0, len(perm), 256):
            xb = W[perm[i:i + 256]]
            # Train AE1 (官方 90-94 行: loss1.backward -> opt1.step -> zero)
            w1, _, w3 = fwd(xb)
            loss1 = (1.0 / n) * ((xb - w1) ** 2).mean() + (1 - 1.0 / n) * ((xb - w3) ** 2).mean()
            loss1.backward(); opt1.step(); opt1.zero_grad()
            # Train AE2 (官方 97-101 行: 重新前向, loss2.backward -> opt2.step)
            _, w2, w3 = fwd(xb)
            loss2 = (1.0 / n) * ((xb - w2) ** 2).mean() - (1 - 1.0 / n) * ((xb - w3) ** 2).mean()
            loss2.backward(); opt2.step(); opt2.zero_grad()
    for m in (E, D1, D2):
        m.eval()

    def score(insts):
        with torch.no_grad():
            F = _mm(torch.from_numpy(insts.reshape(-1, insts.shape[-1] * l)))
            z = E(F); w1 = D1(z); w2 = D2(z)
            return (0.5 * ((F - w1) ** 2).mean(1) + 0.5 * ((F - w2) ** 2).mean(1)).numpy()
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
