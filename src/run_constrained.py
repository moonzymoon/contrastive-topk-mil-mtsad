"""S1补实验: 5基线 + CMH-MIL 在 FAR<=0.1 约束下的 event-F1 (4数据集).
也补 Q3 的 k选择协议验证(k在弱训练+校准选,不触测试)."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.repro import load_config, set_seed
from data.loaders import load_dataset
from data.split import chronological_split, make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss import CMHLoss
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all
from baselines import BASELINES
from run_rebuttal import _build_run

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["SMD", "PSM", "MSL", "SMAP"]


def constrained_event_f1(scores, tY, far_limit=0.1):
    """扫阈值, 取FAR<=limit内最大event-F1."""
    nt = (tY == 0); best = 0.0
    for q in np.linspace(0.01, 0.99, 300):
        thr = float(np.quantile(scores, q))
        pred = (scores > thr).astype(int)
        far = float(pred[nt].mean()) if nt.sum() > 0 else 1.0
        if far <= far_limit:
            m = evaluate_all(tY, scores, thr, events_from_binary(tY), events_from_binary(pred))
            if m["EventF1_one2one"] > best:
                best = m["EventF1_one2one"]
    return best


def baseline_scores(X, Y, cfg, name):
    """获取基线的测试块score(用同一协议)."""
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    cX, cY = bags_to_array(mk(sp.calib)); tX, tY = bags_to_array(mk(sp.test))
    # 各基线 score (重实现核心, 返回 bag 级 score)
    if name == "PCA":
        from sklearn.decomposition import PCA
        wXb, _ = bags_to_array(mk(sp.weak_train)); wF = wXb.reshape(len(wXb), -1)
        pca = PCA(n_components=0.9, svd_solver="full").fit(wF)
        def sc(Xb): F = Xb.reshape(len(Xb), -1); recon = pca.inverse_transform(pca.transform(F)); return ((F-recon)**2).mean(1)
    elif name == "IF":
        from sklearn.ensemble import IsolationForest
        wXb, _ = bags_to_array(mk(sp.weak_train))
        wS = np.concatenate([wXb.mean(axis=(1,2)), wXb.std(axis=(1,2))], axis=1)
        ifo = IsolationForest(n_estimators=100, random_state=7, contamination="auto").fit(wS)
        def sc(Xb): S = np.concatenate([Xb.mean(axis=(1,2)), Xb.std(axis=(1,2))], axis=1); return -ifo.score_samples(S)
    elif name == "AE":
        import torch.nn as nn
        wXb, _ = bags_to_array(mk(sp.weak_train)); dim = wXb[0].size
        class AE(nn.Module):
            def __init__(s): super().__init__(); s.enc=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,16)); s.dec=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,dim))
            def forward(s,x): return s.dec(s.enc(x))
        wF = torch.from_numpy(wXb.reshape(len(wXb),-1)).float(); m = AE(); opt = torch.optim.Adam(m.parameters(),lr=0.002)
        for ep in range(15):
            for i in range(0,len(wF),128):
                idx=range(i,min(i+128,len(wF))); out=m(wF[idx]); loss=((out-wF[idx])**2).mean(); opt.zero_grad();loss.backward();opt.step()
        m.eval()
        def sc(Xb):
            with torch.no_grad():
                F=torch.from_numpy(Xb.reshape(len(Xb),-1)).float(); return ((m(F)-F)**2).mean(1).numpy()
    elif name in ("DeepSVDD","DMSVDD"):
        k = 1 if name=="DeepSVDD" else 3
        from baselines import baseline_deep_svdd
        # 直接复用 baseline_deep_svdd 但取 score: 重训一次取 tScore
        set_seed(7)
        r = _build_run(X, Y, cfg, 7, k=k)
        return r["tBag"], r["tY"]
    return sc(tX), tY


def main():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    res = {"constrained_eventF1_FAR0p1": {}, "k_selection": {}}
    for ds in DATASETS:
        X, Y = load_dataset(ds); print(f"\n=== {ds} ===")
        row = {}
        # CMH-MIL
        set_seed(7); r = _build_run(X, Y, cfg, 7)
        cmh_cef = constrained_event_f1(r["tBag"], r["tY"], 0.1)
        row["CMH-MIL"] = round(cmh_cef, 3)
        print(f"  CMH-MIL constrained EvtF1={cmh_cef:.3f}")
        # 基线
        for bn in ["PCA", "IF", "AE", "DeepSVDD", "DMSVDD"]:
            try:
                sc, tYb = baseline_scores(X, Y, cfg, bn)
                cef = constrained_event_f1(sc, tYb, 0.1)
                row[bn] = round(cef, 3)
                print(f"  {bn} constrained EvtF1={cef:.3f}")
            except Exception as e:
                row[bn] = None; print(f"  {bn} FAIL {str(e)[:40]}")
        res["constrained_eventF1_FAR0p1"][ds] = row
    # Q3: k选择协议 — k在弱训练+校准选(不触测试), 验证测试块结果
    print("\n=== Q3 k选择协议(SMAP, 在校准块选k) ===")
    X, Y = load_dataset("SMAP")
    sp = chronological_split(len(X))
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    cX, cY = bags_to_array(mk(sp.calib))
    cal_auprc = {}
    for k in [1, 2, 3, 5, 8]:
        # 用校准块AUPRC作为k的选择依据(不触测试标签)
        set_seed(7); r = _build_run(X, Y, cfg, 7, k=k)
        # 校准块上算AUPRC(用bag score vs cY)
        from sklearn.metrics import average_precision_score
        try:
            ca = float(average_precision_score(cY, r["cBag"][:len(cY)] if len(r["cBag"])>=len(cY) else r["cBag"]))
        except Exception:
            ca = 0.0
        cal_auprc[k] = round(ca, 3)
        print(f"  k={k}: 校准块AUPRC={ca:.3f} (测试块AUPRC见V-B)")
    res["k_selection"]["SMAP_calib_AUPRC"] = cal_auprc
    out = os.path.join(PROJECT_ROOT, "results", "constrained_all.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
