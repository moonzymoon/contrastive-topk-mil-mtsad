"""基线方法 (第一篇8种 + DeepSVDD + DMSVDD), 同一防泄漏协议评估.
轻量实现, CPU友好. 用于 Table I 对比."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data.split import chronological_split, make_bags, bags_to_array
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all


def _bag_features(bags):
    """bag 内 instance 展平为特征向量 (用于经典/重构基线)."""
    Xb, _ = bags_to_array(bags) if isinstance(bags, list) else (bags, None)
    # 每个 bag: (n,l,D) -> 展平为 mean+std 特征 (轻量)
    return Xb.mean(axis=(1, 2)), Xb.std(axis=(1, 2))  # (N,),(N,) 简化


def baseline_pca(X, Y, cfg):
    """PCA 重构误差 (第一篇基线). bag级重构均方误差作为异常分数."""
    from sklearn.decomposition import PCA
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wXb, wYb = bags_to_array(mk(sp.weak_train))
    # 用弱训练块训练 PCA (保留90%方差), bag展平
    wF = wXb.reshape(len(wXb), -1)
    pca = PCA(n_components=0.9, svd_solver='full').fit(wF)
    def score(bags):
        Xb, Yb = bags_to_array(bags)
        F = Xb.reshape(len(Xb), -1)
        recon = pca.inverse_transform(pca.transform(F))
        err = ((F - recon) ** 2).mean(axis=1)  # 重构误差
        return err, Yb
    cF, cY = score(mk(sp.calib)); tF, tY = score(mk(sp.test))
    thr = split_conformal_threshold(cF[cY == 0], alpha=cfg["conformal"]["alpha"])
    t_pred = (tF > thr).astype(int)
    gt_ev = events_from_binary(tY); pr_ev = events_from_binary(t_pred)
    return evaluate_all(tY, tF, thr, gt_ev, pr_ev)


def baseline_isolation_forest(X, Y, cfg):
    """Isolation Forest (第一篇基线, SMD上反超MIL). bag统计量作为特征."""
    from sklearn.ensemble import IsolationForest
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    def stat(bags):
        Xb, Yb = bags_to_array(bags)
        # 每 bag: (n,l,D) -> mean/std over (n,l) -> (D,) 每bag D维特征
        S = np.concatenate([Xb.mean(axis=(1,2)), Xb.std(axis=(1,2))], axis=1)  # (N, 2D)
        return S, Yb
    wS, _ = stat(mk(sp.weak_train))
    iforest = IsolationForest(n_estimators=100, random_state=0, contamination='auto').fit(wS)
    cS, cY = stat(mk(sp.calib)); tS, tY = stat(mk(sp.test))
    cScore = -iforest.score_samples(cS)
    tScore = -iforest.score_samples(tS)
    thr = split_conformal_threshold(cScore[cY == 0], alpha=cfg["conformal"]["alpha"])
    t_pred = (tScore > thr).astype(int)
    gt_ev = events_from_binary(tY); pr_ev = events_from_binary(t_pred)
    return evaluate_all(tY, tScore, thr, gt_ev, pr_ev)


def baseline_deep_svdd(X, Y, cfg, k=1):
    """Deep SVDD 单球 (导师C1基线). 复用 CMHMIL k=1."""
    import torch
    from models import CMHMIL
    from losses.cmh_loss import CMHLoss
    from utils.repro import set_seed
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wX, wY = bags_to_array(mk(sp.weak_train)); cX, cY = bags_to_array(mk(sp.calib)); tX, tY = bags_to_array(mk(sp.test))
    D = X.shape[1]
    model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"], k=k,
                   channels=tuple(cfg["model"]["encoder"]["channels"]), use_var_attn=cfg["model"]["var_attention"],
                   pool=cfg["conformal"]["bag_pool"])
    lc = cfg["loss"]
    crit = CMHLoss(w_compact=lc["compact"], w_sep=lc["sep"], w_balance=0)  # DeepSVDD无bag标签,用紧致
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    wXt = torch.from_numpy(wX); wYt = torch.from_numpy(wY)
    model.init_centers_kmeans(wXt[torch.randperm(len(wXt))[:min(256,len(wXt))]])
    bs = cfg["train"]["batch_size"]
    for epoch in range(cfg["train"]["max_epochs"]):
        perm = torch.randperm(len(wXt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            scores, _, z = model(wXt[idx], return_z=True)
            loss, _ = crit(scores, wYt[idx], z.reshape(-1, z.shape[-1]), model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"]); opt.step()
    model.eval()
    with torch.no_grad():
        cs, _ = model(torch.from_numpy(cX)); ts, _ = model(torch.from_numpy(tX))
    def pool(arr):
        w = np.exp(arr-arr.max(1,keepdims=True)); w/=w.sum(1,keepdims=True); return (w*arr).sum(1)
    cBag, tBag = pool(cs.numpy()), pool(ts.numpy())
    thr = split_conformal_threshold(cBag[cY==0], alpha=cfg["conformal"]["alpha"])
    t_pred = (tBag > thr).astype(int)
    return evaluate_all(tY, tBag, thr, events_from_binary(tY), events_from_binary(t_pred))


def baseline_autoencoder(X, Y, cfg):
    """重构型 AE 基线 (第一篇 TCAE 类). bag展平重构误差作异常分数."""
    import torch, torch.nn as nn
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wXb, wYb = bags_to_array(mk(sp.weak_train))
    cXb, cY = bags_to_array(mk(sp.calib)); tXb, tY = bags_to_array(mk(sp.test))
    dim = wXb[0].size
    # 轻量 MLP AE
    class AE(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16))
            s.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
        def forward(s, x): return s.dec(s.enc(x))
    wF = torch.from_numpy(wXb.reshape(len(wXb), -1)).float()
    model = AE(dim); opt = torch.optim.Adam(model.parameters(), lr=0.002)
    for ep in range(15):
        perm = torch.randperm(len(wF))
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]; out = model(wF[idx]); loss = ((out-wF[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        def err(Xb):
            F = torch.from_numpy(Xb.reshape(len(Xb), -1)).float()
            return ((model(F)-F)**2).mean(1).numpy()
        cScore = err(cXb); tScore = err(tXb)
    thr = split_conformal_threshold(cScore[cY==0], alpha=cfg["conformal"]["alpha"])
    t_pred = (tScore > thr).astype(int)
    return evaluate_all(tY, tScore, thr, events_from_binary(tY), events_from_binary(t_pred))


BASELINES = {
    "PCA": baseline_pca,
    "IF": baseline_isolation_forest,
    "AE": baseline_autoencoder,
    "DeepSVDD": lambda X,Y,c: baseline_deep_svdd(X,Y,c,k=1),
    "DMSVDD": lambda X,Y,c: baseline_deep_svdd(X,Y,c,k=3),
}


if __name__ == "__main__":
    from utils.repro import load_config
    import json
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data.loaders import load_smd
    X, Y = load_smd("machine-1-1")
    results = {}
    for name, fn in BASELINES.items():
        try:
            m = fn(X, Y, cfg)
            results[name] = m
            print(f"{name}: AUPRC={m['AUPRC']:.3f} EventF1={m['EventF1_one2one']:.3f} PointF1={m['PointF1']:.3f}")
        except Exception as e:
            print(f"{name}: FAIL {e}")
            results[name] = {"error": str(e)}
    out = os.path.join(PROJECT_ROOT, "results", "baselines_SMD_machine-1-1.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved {out}")
