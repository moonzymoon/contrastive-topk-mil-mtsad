"""图件令配套: 从 v3 (MSL seed7) 导出真实分数分布 → CSV (供 pgfplots 直方图)
零字面量纪律: 图2(b) 的每个数字都来自本导出, 不手填.
输出: results/fig2_scores_msl_seed7.csv + 摘要JSON
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from data.protocol import (four_block_split, Scaler, make_bags_v2,
                           point_scores, threshold_from_calib)
from data.loaders import load_dataset
from models import CMHMIL
from utils.repro import set_seed

L, l, BAG_STRIDE = 128, 16, 64


def export(ds="MSL", seed=7):
    set_seed(seed)
    X, Y = load_dataset(ds)
    sp = four_block_split(len(Y))
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)
    D = X.shape[1]

    model = CMHMIL(D, d_embed=24, k=3, channels=(32, 64), kernel_size=3,
                   dropout=0.1, use_var_attn=False, pool="max",
                   radius_mode="quantile", radius_q=0.9)
    # 与 run_v3 相同的训练循环 (复刻, 保证一致性)
    from losses.cmh_loss_v3 import CMHLossV3
    from utils.repro import load_config
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    blocks = {}
    for name in ("train", "calib", "test"):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        blocks[name] = (insts, y_bag, starts + s, rel)
    crit = CMHLossV3(w_contrast=0.5, radius_mode="quantile")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    wX = torch.from_numpy(blocks["train"][0])
    wY = torch.from_numpy(blocks["train"][1].astype(np.float32))
    model.init_centers_kmeans(wX[torch.randperm(len(wX))[:256]])
    bs = cfg["train"]["batch_size"]
    for ep in range(cfg["train"]["max_epochs"]):
        perm = torch.randperm(len(wX))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            scores, bag_score, z = model(wX[idx], return_z=True)
            bag_z = z.mean(dim=1)
            loss, _ = crit(bag_score, wY[idx], scores, z.reshape(-1, z.shape[-1]),
                           bag_z, model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model.hypers.update_radii(z.reshape(-1, z.shape[-1]).detach())

    model.eval()
    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    with torch.no_grad():
        for name in ("calib", "test"):
            insts, yb, starts, rel = blocks[name]
            sc = np.concatenate([
                model(torch.from_numpy(insts[i:i + 256]))[0].numpy()
                for i in range(0, len(insts), 256)])
            s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
            s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
    # 测试块分数与标签 (抽样子采样到5000点供直方图)
    ts, te = sp["test"]
    s_t = s_all[ts:te]; y_t = Y[ts:te]
    cov = ~np.isnan(s_t)
    s_t, y_t = s_t[cov], y_t[cov]
    rng = np.random.default_rng(7)
    idx = rng.choice(len(s_t), min(5000, len(s_t)), replace=False)
    out_csv = os.path.join(os.path.dirname(__file__), "..", "results",
                           f"fig2_scores_{ds}_seed{seed}.csv")
    with open(out_csv, "w") as f:
        f.write("score,label\n")
        for s, y in zip(s_t[idx], y_t[idx]):
            f.write(f"{s:.6f},{int(y)}\n")
    summary = {"threshold": float(thr),
               "n_normal": int((y_t == 0).sum()), "n_anom": int((y_t == 1).sum()),
               "radii": model.hypers.radii.detach().numpy().tolist(),
               "csv": os.path.basename(out_csv)}
    json.dump(summary, open(out_csv.replace(".csv", "_summary.json"), "w"), indent=2)
    print("导出完成:", out_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    export()
