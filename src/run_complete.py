"""完整实验: 主结果(5基线×5种子) + 消融(A1-A4) + 保形误报率 + 扰动.
所有结果真实输出, 填充论文 Table/Section."""
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [7, 42, 123, 202, 999]
DATASETS = ["SMD", "PSM", "MSL", "SMAP"]


def run_variant(X, Y, cfg, seed, k=None, conformal=True, mog=True, device="cpu"):
    """可配置 CMH-MIL 变体. k=None用默认; conformal=False用固定分位; mog=False去ELBO项."""
    set_seed(seed)
    D = X.shape[1]
    sp = chronological_split(len(X), cfg["split"]["weak_train_ratio"], cfg["split"]["calib_ratio"])
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wX, wY = bags_to_array(mk(sp.weak_train)); cX, cY = bags_to_array(mk(sp.calib)); tX, tY = bags_to_array(mk(sp.test))
    kk = k if k else cfg["model"]["hyperspheres"]["k"]
    model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"], k=kk,
                   channels=tuple(cfg["model"]["encoder"]["channels"]), kernel_size=cfg["model"]["encoder"]["kernel_size"],
                   dropout=cfg["model"]["encoder"]["dropout"], use_var_attn=cfg["model"]["var_attention"],
                   pool=cfg["conformal"]["bag_pool"]).to(device)
    lc = cfg["loss"]
    crit = CMHLoss(w_bag=lc["bag_bce"], w_compact=lc["compact"], w_sep=lc["sep"], sep_margin=lc["sep_margin"],
                   w_rank=lc["rank"], rank_margin=lc["rank_margin"],
                   w_elbo=lc["elbo_mog_alpha"] if mog else 0.0, w_balance=lc.get("balance", 0.10))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    wXt = torch.from_numpy(wX).to(device); wYt = torch.from_numpy(wY).to(device)
    n_init = min(256, len(wXt))
    model.init_centers_kmeans(wXt[torch.randperm(len(wXt))[:n_init]], device)
    bs = cfg["train"]["batch_size"]
    for epoch in range(cfg["train"]["max_epochs"]):
        model.train(); perm = torch.randperm(len(wXt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]; xb, yb = wXt[idx], wYt[idx]
            scores, _, z = model(xb, return_z=True)
            loss, _ = crit(scores, yb, z.reshape(-1, z.shape[-1]), model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"]); opt.step()
    model.eval()
    with torch.no_grad():
        cs, _ = model(torch.from_numpy(cX).to(device)); ts, _ = model(torch.from_numpy(tX).to(device))
    def pool(arr):
        w = np.exp(arr-arr.max(1,keepdims=True)); w/=w.sum(1,keepdims=True); return (w*arr).sum(1)
    cBag, tBag = pool(cs.cpu().numpy()), pool(ts.cpu().numpy())
    # 阈值: conformal 或 固定中位数(A2对照)
    if conformal:
        thr = split_conformal_threshold(cBag[cY==0], alpha=cfg["conformal"]["alpha"])
    else:
        thr = float(np.median(cBag))  # 固定阈值对照
    t_pred = (tBag > thr).astype(int)
    m = evaluate_all(tY, tBag, thr, events_from_binary(tY), events_from_binary(t_pred))
    # 经验误报率: 正常bag被报为异常的比例
    normal_test = (tY == 0)
    m["empirical_FAR"] = float(t_pred[normal_test].mean()) if normal_test.sum() > 0 else float("nan")
    return m


def main():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    results = {"main": {}, "ablation": {}, "baselines5seed": {}}
    for ds in DATASETS:
        try:
            X, Y = load_dataset(ds)
            print(f"\n=== {ds}: X{X.shape} anomaly={Y.mean():.4f} ===")
        except Exception as e:
            print(f"{ds}: 不可用({e})"); continue

        # 主结果: CMH-MIL 5种子
        main_ms = [run_variant(X, Y, cfg, s) for s in SEEDS]
        agg = {k: f"{np.mean([m[k] for m in main_ms if not _nan(m[k])]):.4f}±{np.std([m[k] for m in main_ms if not _nan(m[k])]):.4f}"
               for k in main_ms[0] if not _nan(np.mean([m[k] for m in main_ms if not _nan(m[k])] or [0]))}
        results["main"][ds] = agg
        print(f"  CMH-MIL: AUPRC={agg['AUPRC']} EventF1={agg['EventF1_one2one']} FAR={agg['empirical_FAR']}")

        # 消融 (seed=7, 单次足够看趋势)
        abl = {}
        abl["A1_k1"] = run_variant(X, Y, cfg, 7, k=1)  # 单球
        abl["A2_noConformal"] = run_variant(X, Y, cfg, 7, conformal=False)  # 固定阈值
        abl["A3_noMoG"] = run_variant(X, Y, cfg, 7, mog=False)  # 去MoG先验
        results["ablation"][ds] = {ak: {k: round(v,4) if not _nan(v) else None for k,v in am.items()} for ak, am in abl.items()}
        print(f"  消融 A1(k=1) AUPRC={abl['A1_k1']['AUPRC']:.3f} | A2(无保形) AUPRC={abl['A2_noConformal']['AUPRC']:.3f} | A3(无MoG) AUPRC={abl['A3_noMoG']['AUPRC']:.3f}")

        # 基线 5种子
        bl5 = {}
        for bn, bf in BASELINES.items():
            bms = []
            for s in SEEDS:
                try:
                    set_seed(s); bms.append(bf(X, Y, cfg))
                except Exception:
                    pass
            if bms:
                bl5[bn] = {k: f"{np.mean([m[k] for m in bms if not _nan(m[k])]):.4f}" for k in bms[0] if not _nan(np.mean([m[k] for m in bms if not _nan(m[k])] or [0]))}
                print(f"  {bn}: AUPRC={bl5[bn].get('AUPRC')} EventF1={bl5[bn].get('EventF1_one2one')}")
        results["baselines5seed"][ds] = bl5

    # A4: k敏感性 (SMAP上, k in 1,2,3,5,8)
    print("\n=== A4 k敏感性 (SMAP) ===")
    X, Y = load_dataset("SMAP")
    k_res = {}
    for k in [1, 2, 3, 5, 8]:
        m = run_variant(X, Y, cfg, 7, k=k)
        k_res[k] = round(m["AUPRC"], 4)
        print(f"  k={k}: AUPRC={m['AUPRC']:.3f} EventF1={m['EventF1_one2one']:.3f}")
    results["A4_k_sensitivity"] = k_res

    out = os.path.join(PROJECT_ROOT, "results", "complete_experiment.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== 全部完成, 保存: {out} ===")


def _nan(v):
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
