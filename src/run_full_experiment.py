"""全量实验: 多数据集 × CMH-MIL + 基线 × 5种子.
一键生成论文 Table I 完整结果. 数据缺失的自动跳过."""
import os, sys, json, glob, traceback
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.repro import load_config, set_seed
from data.loaders import load_dataset
from baselines import BASELINES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["SMD", "PSM", "MSL", "SMAP", "5ECK2022"]
SEEDS = [7, 42, 123, 202, 999]


def run_cmh_mil(X, Y, cfg, seed, device="cpu"):
    """跑 CMH-MIL (复用 train.py 逻辑, 精简)."""
    from data.split import chronological_split, make_bags, bags_to_array
    from models import CMHMIL
    from losses.cmh_loss import CMHLoss
    from evaluation.conformal import split_conformal_threshold, events_from_binary
    from evaluation.metrics import evaluate_all
    set_seed(seed)
    D = X.shape[1]
    sp = chronological_split(len(X), cfg["split"]["weak_train_ratio"], cfg["split"]["calib_ratio"])
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wX, wY = bags_to_array(mk(sp.weak_train)); cX, cY = bags_to_array(mk(sp.calib)); tX, tY = bags_to_array(mk(sp.test))
    model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"], k=cfg["model"]["hyperspheres"]["k"],
                   channels=tuple(cfg["model"]["encoder"]["channels"]), kernel_size=cfg["model"]["encoder"]["kernel_size"],
                   dropout=cfg["model"]["encoder"]["dropout"], use_var_attn=cfg["model"]["var_attention"],
                   pool=cfg["conformal"]["bag_pool"]).to(device)
    lc = cfg["loss"]
    crit = CMHLoss(w_bag=lc["bag_bce"], w_compact=lc["compact"], w_sep=lc["sep"], sep_margin=lc["sep_margin"],
                   w_rank=lc["rank"], rank_margin=lc["rank_margin"], w_elbo=lc["elbo_mog_alpha"], w_balance=lc.get("balance", 0.10))
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
    thr = split_conformal_threshold(cBag[cY==0], alpha=cfg["conformal"]["alpha"])
    t_pred = (tBag > thr).astype(int)
    return evaluate_all(tY, tBag, thr, events_from_binary(tY), events_from_binary(t_pred))


def main():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    all_results = {}
    for ds in DATASETS:
        try:
            X, Y = load_dataset(ds)
            print(f"\n=== {ds}: X{X.shape} anomaly_rate={Y.mean():.4f} ===")
        except Exception as e:
            print(f"\n=== {ds}: 数据不可用({e}), 跳过 ==="); continue
        ds_results = {}
        # CMH-MIL 5种子
        cmh_metrics = []
        for seed in SEEDS:
            try:
                m = run_cmh_mil(X, Y, cfg, seed)
                cmh_metrics.append(m)
                print(f"  CMH-MIL seed{seed}: AUPRC={m['AUPRC']:.3f} EventF1={m['EventF1_one2one']:.3f}")
            except Exception as e:
                print(f"  CMH-MIL seed{seed}: FAIL {e}")
        if cmh_metrics:
            ds_results["CMH-MIL"] = {k: f"{np.mean([m[k] for m in cmh_metrics]):.4f}±{np.std([m[k] for m in cmh_metrics]):.4f}" for k in cmh_metrics[0]}
        # 基线(单种子, 快速)
        for bname, bfn in BASELINES.items():
            try:
                set_seed(7)
                m = bfn(X, Y, cfg)
                ds_results[bname] = {k: f"{v:.4f}" for k, v in m.items()}
                print(f"  {bname}: AUPRC={m['AUPRC']:.3f} EventF1={m['EventF1_one2one']:.3f}")
            except Exception as e:
                print(f"  {bname}: FAIL {e}")
        all_results[ds] = ds_results
    # 保存
    out = os.path.join(PROJECT_ROOT, "results", "full_experiment.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n=== 全量结果保存: {out} ===")


if __name__ == "__main__":
    main()
