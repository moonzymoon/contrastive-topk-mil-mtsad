"""W3 全量重跑 (定位终裁数据源, 2026-11-17)
==========================================
- 11 数据集: SMD PSM MSL SMAP SWaT WADI Exathlon-1 Exathlon-2 Genesis Daphnet CATSv2
- v3 主方法: 5 种子 (7,42,123,202,999), pool=max, radius=quantile b2
- 经典基线 (PCA/IF/AE): 1 次 (确定性/固定seed)
- 神经基线 (DeepSVDD/USAD/MS-SVDD-style): 3 种子
- AT-faithful: 3 种子
- ⑪ SMD-28: 28实体 × v3 × 5种子, 实体名+每实体指标留档
全部走 W1 protocol (数字同源). 输出: results/full_v3_results.json
"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from run_v3 import run_v3, load
from baselines_v3 import BASELINES_V3
from baseline_at_faithful import baseline_at_faithful

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI",
            "Exathlon-1", "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]
SEEDS_V3 = [7, 42, 123, 202, 999]
SEEDS_NN = [7, 42, 123]

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "full_v3_results.json")


def agg(per, keys=("AUPRC", "PointF1", "EventF1_one2one", "FAR_point")):
    a = {}
    for k in keys:
        v = [p[k] for p in per if k in p and p[k] == p[k]]
        if v:
            a[k] = (f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}"
                    if len(v) > 1 else f"{v[0]:.4f}")
    return a


def main():
    results = {}
    if os.path.exists(OUT):   # 断点续跑
        results = json.load(open(OUT, encoding="utf-8"))
        print(f"续跑: 已有 {list(results.keys())}")

    for ds in DATASETS:
        if ds in results:
            continue
        try:
            X, Y = load(ds)
        except Exception as e:
            print(f"[{ds}] 加载失败: {str(e)[:80]}")
            continue
        entry = {"T": len(Y)}
        t0 = time.time()

        # v3 主方法 5 种子
        per = []
        for sd in SEEDS_V3:
            try:
                m = run_v3(X, Y, sd)
                per.append(m)
                print(f"[{ds}] v3 seed{sd}: AUPRC={m['AUPRC']:.4f}")
            except Exception as e:
                print(f"[{ds}] v3 seed{sd} FAIL: {str(e)[:80]}")
        if per:
            entry["ours_v3"] = {"per_seed": per, "aggregate": agg(per)}

        # 经典基线 (1次) + 神经基线 (3种子)
        for name, fn in BASELINES_V3.items():
            n_seeds = 1 if name in ("PCA", "IF") else SEEDS_NN
            per = []
            for sd in (SEEDS_V3[:1] if n_seeds == 1 else SEEDS_NN):
                try:
                    m = fn(X, Y, seed=sd)
                    per.append(m)
                except Exception as e:
                    print(f"[{ds}] {name} seed{sd} FAIL: {str(e)[:80]}")
            if per:
                entry[name] = {"per_seed": per, "aggregate": agg(per)}
            print(f"[{ds}] {name}: {entry.get(name, {}).get('aggregate', {}).get('AUPRC', 'FAIL')}")

        # AT faithful 3 种子
        per = []
        for sd in SEEDS_NN:
            try:
                m = baseline_at_faithful(X, Y, seed=sd)
                per.append(m)
            except Exception as e:
                print(f"[{ds}] AT seed{sd} FAIL: {str(e)[:80]}")
        if per:
            entry["AT-faithful"] = {"per_seed": per, "aggregate": agg(per)}
        print(f"[{ds}] AT-faithful: {entry.get('AT-faithful', {}).get('aggregate', {}).get('AUPRC', 'FAIL')}")

        entry["_seconds"] = round(time.time() - t0)
        results[ds] = entry
        json.dump(results, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"[{ds}] 完成 ({entry['_seconds']}s), 已落盘\n")

    print("=" * 60)
    print("汇总 (AUPRC):")
    for ds, e in results.items():
        row = f"  {ds:12s} v3={e.get('ours_v3', {}).get('aggregate', {}).get('AUPRC', '-'):>14s}"
        for b in ("PCA", "AE", "USAD", "AT-faithful"):
            row += f" {b}={e.get(b, {}).get('aggregate', {}).get('AUPRC', '-'):>14s}"
        print(row)


if __name__ == "__main__":
    main()
