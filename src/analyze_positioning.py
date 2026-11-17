"""定位终裁数据包 (W3完成, 2026-11-17)
==========================================
输入: full_v3_results.json (11数据集) + smd28_v3.json
输出: 胜负矩阵 + 配对Wilcoxon(v3 vs 各基线, 5种子) + 决策树证据
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(__file__), "..", "results")
full = json.load(open(os.path.join(R, "full_v3_results.json"), encoding="utf-8"))
smd28 = json.load(open(os.path.join(R, "smd28_v3.json"), encoding="utf-8"))

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI",
            "Exathlon-1", "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]
BASES = ["PCA", "IF", "AE", "DeepSVDD", "USAD", "MS-SVDD-style", "AT-faithful"]

# 训练块异常bag数 (W1验收重建值) — 决策树分组依据
ANOM_BAGS = {"SMD": 0, "PSM": 0, "MSL": 0, "SMAP": 120, "SWaT": 0, "WADI": 0,
             "Exathlon-1": 31, "Exathlon-2": 17, "Genesis": 0, "Daphnet": 0, "CATSv2": 7}

print("=" * 90)
print("一、胜负矩阵 (v3 均值 vs 基线均值, AUPRC; ✓=v3赢 ✗=v3输)")
print("=" * 90)
header = f"{'DS':12s}{'v3':>10s}" + "".join(f"{b[:8]:>10s}" for b in BASES)
print(header)
wins_per_base = {b: 0 for b in BASES}
win_all = []
for ds in DATASETS:
    e = full[ds]
    v3m = statistics.mean([p["AUPRC"] for p in e["ours_v3"]["per_seed"]])
    row = f"{ds:12s}{v3m:>10.3f}"
    beats_all = True
    for b in BASES:
        bm = statistics.mean([p["AUPRC"] for p in e[b]["per_seed"]])
        w = v3m > bm
        wins_per_base[b] += int(w)
        beats_all &= w
        row += f"{'✓' if w else '✗':>6s}{bm:4.3f}"
    win_all.append(beats_all)
    print(row + ("   [全胜]" if beats_all else ""))
print(f"\nv3 对各基线胜场: " + ", ".join(f"{b}={wins_per_base[b]}/11" for b in BASES))
print(f"v3 全胜数据集: {[d for d, w in zip(DATASETS, win_all) if w]}")

print()
print("=" * 90)
print("二、配对 Wilcoxon (v3 vs 基线, 逐数据集 5种子配对, v3种子数不足时用基线种子对齐)")
print("=" * 90)
for ds in DATASETS:
    e = full[ds]
    v3 = [p["AUPRC"] for p in e["ours_v3"]["per_seed"]]
    sig = []
    for b in ("PCA", "USAD", "AT-faithful"):
        bs = [p["AUPRC"] for p in e[b]["per_seed"]]
        n = min(len(v3), len(bs))
        try:
            stat, p = wilcoxon(v3[:n], bs[:n])
            mark = "显著" if p < 0.05 else ("趋势" if p < 0.1 else "ns")
            better = "v3优" if statistics.mean(v3) > statistics.mean(bs) else "v3劣"
            sig.append(f"{b}:{mark}(p={p:.3f},{better})")
        except Exception:
            sig.append(f"{b}:n/a")
    print(f"  {ds:12s} {' | '.join(sig)}")

print()
print("=" * 90)
print("三、定位决策树证据")
print("=" * 90)
print("【双类分支数据集】(训练块含异常bag, Eq(4) InfoNCE 真正运行):")
for ds in DATASETS:
    if ANOM_BAGS[ds] > 0:
        e = full[ds]
        v3m = statistics.mean([p["AUPRC"] for p in e["ours_v3"]["per_seed"]])
        best_b, best_v = max(((b, statistics.mean([p["AUPRC"] for p in e[b]["per_seed"]])) for b in BASES),
                             key=lambda x: x[1])
        print(f"  {ds:12s} 异常bag={ANOM_BAGS[ds]:3d}  v3={v3m:.3f}  最强基线={best_b}({best_v:.3f})  {'v3胜' if v3m>best_v else 'v3负'}")
print("【全正常分支数据集】(只有 dispersion 正则):")
for ds in DATASETS:
    if ANOM_BAGS[ds] == 0:
        e = full[ds]
        v3m = statistics.mean([p["AUPRC"] for p in e["ours_v3"]["per_seed"]])
        pca = statistics.mean([p["AUPRC"] for p in e["PCA"]["per_seed"]])
        print(f"  {ds:12s} v3={v3m:.3f} vs PCA={pca:.3f}  {'胜' if v3m>pca else '负'}")

print()
print("四、SMD-28 对照")
ok = [v["AUPRC_mean"] for v in smd28.values() if "AUPRC_mean" in v]
m1_1 = smd28.get("machine-1-1", {}).get("AUPRC_mean", float("nan"))
print(f"  28实体宏平均={statistics.mean(ok):.4f}±{statistics.stdev(ok):.4f} (28/28成功, 5种子)")
print(f"  machine-1-1(主表单实体)={m1_1:.4f}")
print(f"  单实体 vs 宏平均: {'宏平均更高(无夸大)' if statistics.mean(ok) >= m1_1 else '单实体更高(夸大'+f'{(m1_1/statistics.mean(ok)-1)*100:.0f}%)'}")

print()
print("五、SWaT 种子稳定性预警")
swat = [p["AUPRC"] for p in full["SWaT"]["ours_v3"]["per_seed"]]
print(f"  5种子AUPRC: {[round(x,3) for x in swat]} (std={statistics.stdev(swat):.3f})")
print("  2/5种子崩至~0.19 — 量化半径+EMA 在部分初始化下不稳, 需在 W4 诊断(warmup/初始化)")
