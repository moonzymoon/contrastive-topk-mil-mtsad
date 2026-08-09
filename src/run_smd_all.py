"""SMD 全 28 实体实验 (第一篇口径: 每实体独立跑, 跨实体聚合).
NASA 航天器同理. 调用 train_one_seed 但传入不同 entity."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.repro import load_config, set_seed, setup_logging
from data.split import chronological_split, make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss import CMHLoss
from evaluation.conformal import split_conformal_threshold, conformal_alarm, events_from_binary
from evaluation.metrics import evaluate_all
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMD_DIR = os.path.join(PROJECT_ROOT, "datasets", "SMD")


def list_smd_entities():
    """列出所有 SMD 实体文件名."""
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(SMD_DIR, "train", "*.txt")))


def run_entity(entity, cfg, seed, device="cpu"):
    """跑单个 SMD 实体, 返回 metrics."""
    import numpy as np
    from data.loaders import _zscore
    Xtr = np.loadtxt(os.path.join(SMD_DIR, "train", f"{entity}.txt"), delimiter=",").astype(np.float32)
    Xte = np.loadtxt(os.path.join(SMD_DIR, "test", f"{entity}.txt"), delimiter=",").astype(np.float32)
    Yte = np.loadtxt(os.path.join(SMD_DIR, "test_label", f"{entity}.txt"), dtype=np.int8)
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    n_tr = len(Xtr) // 2
    Xtr = _zscore(Xtr[:n_tr], Xtr); Xte = _zscore(Xtr[:n_tr], Xte)
    X = np.vstack([Xtr, Xte]); Y = np.concatenate([np.zeros(len(Xtr), np.int8), Yte])
    D = X.shape[1]
    sp = chronological_split(len(X), cfg["split"]["weak_train_ratio"], cfg["split"]["calib_ratio"])
    mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
    wX, wY = bags_to_array(mk(sp.weak_train)); cX, cY = bags_to_array(mk(sp.calib)); tX, tY = bags_to_array(mk(sp.test))

    set_seed(seed)
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
    # 校准+测试
    model.eval()
    with torch.no_grad():
        cs, _ = model(torch.from_numpy(cX).to(device)); ts, _ = model(torch.from_numpy(tX).to(device))
    def pool_fn(arr):
        if cfg["conformal"]["bag_pool"] == "softmax":
            w = np.exp(arr - arr.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True); return (w * arr).sum(1)
        return arr.mean(1)
    c_bag = pool_fn(cs.cpu().numpy()); t_bag = pool_fn(ts.cpu().numpy())
    thr = split_conformal_threshold(c_bag[cY == 0], alpha=cfg["conformal"]["alpha"])
    t_pred = (t_bag > thr).astype(int)
    gt_ev = events_from_binary(tY); pr_ev = events_from_binary(t_pred)
    return evaluate_all(tY, t_bag, thr, gt_ev, pr_ev)


def main():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    entities = list_smd_entities()
    seeds = [7]  # 先单种子跑全部28实体评估可行性
    print(f"SMD 实体数: {len(entities)}, 种子: {seeds}")
    all_m = []
    for ei, ent in enumerate(entities):
        try:
            m = run_entity(ent, cfg, seeds[0])
            all_m.append(m)
            print(f"[{ei+1}/{len(entities)}] {ent}: AUPRC={m['AUPRC']:.3f} EventF1={m['EventF1_one2one']:.3f}")
        except Exception as e:
            print(f"[{ei+1}] {ent}: FAIL {e}")
    # 跨实体聚合(第一篇: 28实体作为SMD的独立单元)
    print("\n===== SMD 跨28实体聚合 (seed={}) =====".format(seeds[0]))
    for metric in ["AUPRC", "EventF1_one2one", "PointF1", "AllEventMIoU", "VUS_PR"]:
        vals = [m[metric] for m in all_m if not np.isnan(m[metric])]
        print(f"{metric}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} median={np.median(vals):.4f}")
    # 保存
    out = os.path.join(PROJECT_ROOT, "results", "SMD_28entities_seed7.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"entities": entities, "seed": seeds[0], "per_entity": all_m}, f, ensure_ascii=False, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
