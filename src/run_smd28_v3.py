"""⑪ SMD-28 重跑 (W3, 2026-11-17): 28实体 × v3(quantile半径) × 5种子
实体名 + 每实体每种子指标留档 (旧版27条无名记录作废).
输出: results/smd28_v3.json
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data.loaders import load_smd
from data.protocol import four_block_split, Scaler, make_bags_v2
from run_v3 import run_v3

SEEDS = [7, 42, 123, 202, 999]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "smd28_v3.json")


def main():
    entities = [f"machine-{i}-{j}" for i in (1, 2, 3) for j in range(1, 9)]
    entities += [f"machine-3-{j}" for j in (9, 10, 11)]
    # OmniAnomaly 命名: machine-1-1..1-8, 2-1..2-9, 3-1..3-11 = 28
    entities = ([f"machine-1-{j}" for j in range(1, 9)]
                + [f"machine-2-{j}" for j in range(1, 10)]
                + [f"machine-3-{j}" for j in range(1, 12)])
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT, encoding="utf-8"))
    print(f"共 {len(entities)} 实体, 已有 {len(results)}")

    for ent in entities:
        if ent in results:
            continue
        try:
            X, Y = load_smd(ent)
        except Exception as e:
            results[ent] = {"error": str(e)[:150]}
            print(f"[{ent}] 加载失败: {str(e)[:60]}")
            continue
        per = []
        for sd in SEEDS:
            try:
                m = run_v3(X, Y, sd)
                per.append(m)
            except Exception as e:
                print(f"[{ent}] seed{sd} FAIL: {str(e)[:60]}")
        if per:
            auprcs = [p["AUPRC"] for p in per]
            results[ent] = {
                "per_seed": per,
                "AUPRC_mean": statistics.mean(auprcs),
                "AUPRC_std": statistics.stdev(auprcs) if len(auprcs) > 1 else 0,
            }
            print(f"[{ent}] AUPRC={results[ent]['AUPRC_mean']:.4f}±{results[ent]['AUPRC_std']:.4f} (n={len(per)})")
        json.dump(results, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)

    ok = [v["AUPRC_mean"] for v in results.values() if "AUPRC_mean" in v]
    errs = [k for k, v in results.items() if "error" in v]
    if ok:
        print("=" * 60)
        print(f"SMD-28 v3: {len(ok)} 实体成功, {len(errs)} 失败 {errs}")
        print(f"宏平均 AUPRC = {statistics.mean(ok):.4f} ± {statistics.stdev(ok):.4f}")


if __name__ == "__main__":
    main()
