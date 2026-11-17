"""W1 验收脚本 (2026-11-17)
==========================================
A. 各数据集在新协议下的 bag 统计重建 (对照 GPT 外审探针表:
   SMD 310/0, PSM 1203/0, MSL 721/0, SMAP 3076/120,
   Exathlon-1 234/31, Exathlon-2 254/17, Genesis 87/0, Daphnet 211/0, CATSv2 545/7)
B. PCA 基线接入新协议 (点级) 在 SMD 端到端跑通 — W1 完成演示
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data.protocol import (four_block_split, Scaler, make_bags_v2,
                           point_scores, threshold_from_calib, evaluate_point)
from data.loaders import load_dataset
from data.tsb_ad_loader import load_tsb_ad

L, l, bag_stride = 128, 16, 64

# GPT 外审探针的旧协议统计 (bag 数与 instance 构造无关, 应能对上)
GPT_TABLE = {
    "SMD": (310, 0), "PSM": (1203, 0), "MSL": (721, 0), "SMAP": (3076, 120),
    "Exathlon-1": (234, 31), "Exathlon-2": (254, 17),
    "Genesis": (87, 0), "Daphnet": (211, 0), "CATSv2": (545, 7),
}


def load(ds):
    if ds.startswith("Exathlon"):
        return load_tsb_ad("Exathlon", int(ds.split("-")[1]))
    if ds in ("Genesis", "Daphnet", "CATSv2"):
        return load_tsb_ad(ds, 1)
    return load_dataset(ds)


def bag_stats(ds, X, Y):
    """新协议 train 块的 bag 统计."""
    sp = four_block_split(len(Y))
    s, e = sp["train"]
    _, y_bag, _, _ = make_bags_v2(X[s:e], Y[s:e], L, l, bag_stride)
    return len(y_bag), int(y_bag.sum())


def pca_point_baseline(X, Y):
    """PCA 重构误差基线, 完整走 W1 新协议 (点级).
    特征: instance 原始值展平; 分数: 重构MSE; 点级max聚合; calib阈值; 点级指标."""
    from sklearn.decomposition import PCA
    sp = four_block_split(len(Y))
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)

    blocks = {}
    for name in ("train", "calib", "test"):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, bag_stride)
        blocks[name] = (insts, y_bag, starts + s, rel)  # starts 映射回全流坐标

    tr = blocks["train"][0].reshape(-1, blocks["train"][0].shape[-1] * l)  # (N*n, l*D) 每 instance 一行
    pca = PCA(n_components=0.9, svd_solver="full").fit(tr)

    def inst_scores(insts):
        N, n = insts.shape[0], insts.shape[1]
        F = insts.reshape(N * n, -1)                     # (N*n, l*D)
        recon = pca.inverse_transform(pca.transform(F))
        err = ((F - recon) ** 2).mean(axis=1)            # (N*n,)
        return err.reshape(N, n)                          # (N, n) 每 instance 一分

    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    for name in ("calib", "test"):
        insts, yb, starts, rel = blocks[name]
        sc = inst_scores(insts)
        s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
        # 只填本块区间 (point_scores 以全流坐标输出)
        s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=0.05)
    cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                       s_all[sp["test"][0]:sp["test"][1]], thr, cov)
    m["threshold"] = thr
    return m


if __name__ == "__main__":
    print("=" * 70)
    print("A. 新协议 bag 统计重建 (对照 GPT 探针表)")
    print("=" * 70)
    report = {}
    for ds, (gpt_n, gpt_a) in GPT_TABLE.items():
        try:
            X, Y = load(ds)
            n, a = bag_stats(ds, X, Y)
            match = "✓" if abs(n - gpt_n) <= 1 and a == gpt_a else f"✗(GPT:{gpt_n}/{gpt_a})"
            print(f"  {ds:12s} T={len(Y):7d}  train_bags={n:5d} anom_bags={a:4d}  {match}")
            report[ds] = {"T": len(Y), "train_bags": n, "anom_bags": a, "gpt": [gpt_n, gpt_a]}
        except Exception as ex:
            print(f"  {ds:12s} 加载失败: {str(ex)[:60]}")
            report[ds] = {"error": str(ex)[:200]}

    print()
    print("=" * 70)
    print("B. PCA 走新协议端到端 (SMD, 点级)")
    print("=" * 70)
    X, Y = load("SMD")
    m = pca_point_baseline(X, Y)
    for k in ("AUPRC", "AUROC", "PointF1", "EventF1_one2one", "FAR_point", "coverage", "threshold"):
        print(f"  {k:16s} {m[k]}")
    report["PCA_SMD_pointlevel"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                    for k, v in m.items()}

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "w1_validation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已存: {out}")
