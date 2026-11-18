"""CMH-MIL v3 主实验脚本 (W1协议 + W2损失 重建版, 2026-11-17)
==============================================================
接入:
  W1: data.protocol 四块切分/先切分后标准化/make_bags_v2(n=8全覆盖)/点级评测
  W2: losses.cmh_loss_v3 (⑥InfoNCE论文式 ⑨dispersion诚实命名 ⑦bagBCE解耦)
参数化 (供 W2 报批快跑):
  --pool {max,softmax}   ⑦ 对照 (max=any-point语义 vs softmax=平滑)
  --var_attn {0,1}       ⑧ 对照 (0=删除死代码组件, 默认; 1=预留)
用法:
  python run_v3.py --dataset MSL --seeds 7,42,123 --pool max
"""
import os, sys, json, time, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from data.protocol import (four_block_split, official_split, weak_annotation_split,
                           Scaler, make_bags_v2,
                           point_scores, threshold_from_calib, evaluate_point)
from data.loaders import load_dataset
from data.tsb_ad_loader import load_tsb_ad
from models import CMHMIL
from losses.cmh_loss_v3 import CMHLossV3
from utils.repro import set_seed

L, l, BAG_STRIDE = 128, 16, 64


def load(ds):
    if ds.startswith("Exathlon"):
        return load_tsb_ad("Exathlon", int(ds.split("-")[1]))
    if ds in ("Genesis", "Daphnet", "CATSv2"):
        return load_tsb_ad(ds, 1)
    return load_dataset(ds)  # (X, Y, n_official_train) 方案C三元组


def run_v3(X, Y, seed, pool="max", var_attn=False, epochs=None, lr=2e-3,
           w_contrast=0.5, w_radius=0.1, clip_percentile=None, log=print,
           radius_mode="quantile", radius_q=0.9,
           split_mode="four-block", n_official_train=None, bag_ce_only=False,
           freeze_epochs=0):
    """单种子完整管线: W1协议 → 模型 → v3损失训练 → instance分数 → 点级评测.

    方案C扩展 (2026-11-18):
      split_mode: 'four-block'   旧全流 35/10/10/45 (历史兼容)
                  'plan-c'       轨1 official_split (annot 不消费)
                  'plan-c-annot' 轨2 weak_annotation_split (annot bag 并入训练)
      n_official_train: loaders 三元组第三项 (轨1/轨2必传)
      bag_ce_only: R2-4 消融 — w_contrast=0 且 w_topk=0 (仅 bag BCE + 空间正则)
    """
    from utils.repro import load_config
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    if epochs is None:
        epochs = cfg["train"]["max_epochs"]
    set_seed(seed)
    if split_mode == "four-block":
        sp = four_block_split(len(Y))
    elif split_mode == "plan-c":
        sp = official_split(len(Y), n_official_train)
    elif split_mode == "plan-c-annot":
        sp = weak_annotation_split(len(Y), n_official_train)
    else:
        raise ValueError(split_mode)
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]], clip_percentile=clip_percentile)
    Xs = scaler.transform(X)
    D = X.shape[1]

    blocks = {}
    for name in ("train", "calib", "test"):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        blocks[name] = (insts, y_bag, starts + s, rel)

    # 轨2: annot 块 bag 并入训练 (同预算消费弱标签; 不参与校准/评估)
    if split_mode == "plan-c-annot":
        s, e = sp["annot"]
        a_insts, a_y, _, _ = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        wX_np = np.concatenate([blocks["train"][0], a_insts])
        wY_np = np.concatenate([blocks["train"][1], a_y])
        n_annot_bags = len(a_y)
    else:
        wX_np, wY_np = blocks["train"][0], blocks["train"][1]
        n_annot_bags = 0

    model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"],
                   k=cfg["model"]["hyperspheres"]["k"],
                   channels=tuple(cfg["model"]["encoder"]["channels"]),
                   kernel_size=cfg["model"]["encoder"]["kernel_size"],
                   dropout=cfg["model"]["encoder"]["dropout"],
                   use_var_attn=var_attn, pool=pool,   # ⑦⑧ 参数化
                   radius_mode=radius_mode, radius_q=radius_q)  # ⑤b2
    if bag_ce_only:
        crit = CMHLossV3(w_contrast=0.0, w_topk=0.0, w_radius=w_radius,
                         radius_mode=radius_mode)
    else:
        crit = CMHLossV3(w_contrast=w_contrast, w_radius=w_radius,
                         radius_mode=radius_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg["train"]["weight_decay"])

    wX = torch.from_numpy(wX_np)
    wY = torch.from_numpy(wY_np.astype(np.float32))
    model.init_centers_kmeans(wX[torch.randperm(len(wX))[:min(256, len(wX))]])
    init_centers = model.hypers.centers.detach().clone()  # SWaT warmup: 冻结基准
    bs = cfg["train"]["batch_size"]

    model.train()
    for ep in range(epochs):
        frozen = ep < freeze_epochs  # SWaT 诊断: 前freeze_epochs轮冻结中心+半径
        perm = torch.randperm(len(wX))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            xb, yb = wX[idx], wY[idx]
            scores, bag_score, z = model(xb, return_z=True)
            bag_z = z.mean(dim=1)  # bag 池化表征 (对比学习输入)
            loss, parts = crit(bag_score, yb, scores, z.reshape(-1, z.shape[-1]),
                               bag_z, model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            if frozen:
                # 冻结: 恢复中心与半径到初始值 (只训 encoder/表征)
                with torch.no_grad():
                    model.hypers.centers.copy_(init_centers)
            else:
                # ⑤b2: 每步 EMA 分位数半径更新 (无梯度统计量)
                model.hypers.update_radii(z.reshape(-1, z.shape[-1]).detach())

    # ---- 推理: instance 分数 → 点级 ----
    model.eval()
    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    with torch.no_grad():
        for name in ("calib", "test"):
            insts, yb, starts, rel = blocks[name]
            sc = []
            for i in range(0, len(insts), 256):
                s_i, _ = model(torch.from_numpy(insts[i:i + 256]))
                sc.append(s_i.numpy())
            sc = np.concatenate(sc)  # (N, n)
            s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
            s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
    cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                       s_all[sp["test"][0]:sp["test"][1]], thr, cov)
    m["threshold"] = thr
    m["train_bags"] = int(len(wY_np))
    m["train_anom_bags"] = int(wY_np.sum())
    m["annot_bags"] = int(n_annot_bags)
    m["split_mode"] = split_mode
    m["bag_ce_only"] = bool(bag_ce_only)
    m["final_radii"] = model.hypers.radii.detach().numpy().tolist()  # ⑤ 监控
    return m

    model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"],
                   k=cfg["model"]["hyperspheres"]["k"],
                   channels=tuple(cfg["model"]["encoder"]["channels"]),
                   kernel_size=cfg["model"]["encoder"]["kernel_size"],
                   dropout=cfg["model"]["encoder"]["dropout"],
                   use_var_attn=var_attn, pool=pool,   # ⑦⑧ 参数化
                   radius_mode=radius_mode, radius_q=radius_q)  # ⑤b2
    crit = CMHLossV3(w_contrast=w_contrast, w_radius=w_radius,
                     radius_mode=radius_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg["train"]["weight_decay"])

    wX = torch.from_numpy(blocks["train"][0])
    wY = torch.from_numpy(blocks["train"][1].astype(np.float32))
    model.init_centers_kmeans(wX[torch.randperm(len(wX))[:min(256, len(wX))]])
    bs = cfg["train"]["batch_size"]

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(wX))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            xb, yb = wX[idx], wY[idx]
            scores, bag_score, z = model(xb, return_z=True)
            bag_z = z.mean(dim=1)  # bag 池化表征 (对比学习输入)
            loss, parts = crit(bag_score, yb, scores, z.reshape(-1, z.shape[-1]),
                               bag_z, model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            # ⑤b2: 每步 EMA 分位数半径更新 (无梯度统计量)
            model.hypers.update_radii(z.reshape(-1, z.shape[-1]).detach())

    # ---- 推理: instance 分数 → 点级 ----
    model.eval()
    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    with torch.no_grad():
        for name in ("calib", "test"):
            insts, yb, starts, rel = blocks[name]
            sc = []
            for i in range(0, len(insts), 256):
                s_i, _ = model(torch.from_numpy(insts[i:i + 256]))
                sc.append(s_i.numpy())
            sc = np.concatenate(sc)  # (N, n)
            s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
            s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
    cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                       s_all[sp["test"][0]:sp["test"][1]], thr, cov)
    m["threshold"] = thr
    m["train_bags"] = int(len(blocks["train"][1]))
    m["train_anom_bags"] = int(blocks["train"][1].sum())
    m["final_radii"] = model.hypers.radii.detach().numpy().tolist()  # ⑤ 监控
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MSL")
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--pool", default="max", choices=["max", "softmax"])
    ap.add_argument("--radius_mode", default="quantile", choices=["quantile", "learnable"])
    ap.add_argument("--radius_q", type=float, default=0.9)
    ap.add_argument("--var_attn", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split_mode", default="four-block",
                    choices=["four-block", "plan-c", "plan-c-annot"])
    ap.add_argument("--bag_ce_only", type=int, default=0)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    X, Y, n_official_train = load(a.dataset)
    print(f"== {a.dataset} T={len(Y)} pool={a.pool} var_attn={a.var_attn} "
          f"split={a.split_mode} bagCE={a.bag_ce_only} seeds={seeds}")
    per = []
    for sd in seeds:
        t0 = time.time()
        m = run_v3(X, Y, sd, pool=a.pool, var_attn=bool(a.var_attn), epochs=a.epochs,
                  radius_mode=a.radius_mode, radius_q=a.radius_q,
                  split_mode=a.split_mode, n_official_train=n_official_train,
                  bag_ce_only=bool(a.bag_ce_only))
        per.append(m)
        print(f"  seed {sd}: AUPRC={m['AUPRC']:.4f} PF1={m['PointF1']:.4f} "
              f"EF1={m['EventF1_one2one']:.4f} FAR={m['FAR_point']:.4f} "
              f"radii={[round(r,2) for r in m['final_radii']]} ({time.time()-t0:.0f}s)")
    agg = {}
    for k in ("AUPRC", "PointF1", "EventF1_one2one", "FAR_point"):
        v = [p[k] for p in per]
        agg[k] = f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
    print("  AGG:", agg)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"v3_{a.dataset}_pool-{a.pool}_attn{a.var_attn}{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": a.dataset, "pool": a.pool, "var_attn": a.var_attn,
                   "per_seed": per, "aggregate": agg}, f, ensure_ascii=False, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
