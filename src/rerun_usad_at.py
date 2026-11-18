"""R4: USAD/AT 补跑（修后推理公式: USAD D2(E(w1)) / AT 逐点→instance级）
在 run_full_v3_c.py 完成后运行：删除 JSON 中的旧 USAD/AT 条目，用修后代码重跑这两列.
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from run_v3 import load
from baselines_v3 import baseline_usad
from baseline_at_faithful import baseline_at_faithful

SEEDS = [7, 42, 123, 202, 999]
R = os.path.join(os.path.dirname(__file__), "..", "results")
T1 = os.path.join(R, "full_v3c_track1.json")

def agg(per):
    a = {}
    for k in ("AUPRC", "AUROC", "PointF1", "EventF1_one2one", "FAR_point"):
        v = [p[k] for p in per if k in per and p[k] == p[k]]
        if v:
            a[k] = f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
    return a

t1 = json.load(open(T1, encoding="utf-8"))
for ds in t1:
    # 删旧USAD/AT
    changed = False
    for m in ("USAD", "AT-faithful"):
        if m in t1[ds]:
            del t1[ds][m]
            changed = True
    if not changed:
        print(f"[{ds}] 无旧USAD/AT, 跳过")
        continue

    X, Y, n_off = load(ds)
    print(f"\n=== [{ds}] USAD/AT 补跑 (修后公式) ===")

    per_usad = []
    for sd in SEEDS:
        m = baseline_usad(X, Y, seed=sd, n_official_train=n_off)
        m["seed"] = sd
        per_usad.append(m)
    t1[ds]["USAD"] = {"per_seed": per_usad, "aggregate": agg(per_usad)}
    print(f"  USAD: {t1[ds]['USAD']['aggregate'].get('AUPRC')}")

    per_at = []
    for sd in SEEDS:
        m = baseline_at_faithful(X, Y, seed=sd, n_official_train=n_off)
        m["seed"] = sd
        per_at.append(m)
    t1[ds]["AT-faithful"] = {"per_seed": per_at, "aggregate": agg(per_at)}
    print(f"  AT: {t1[ds]['AT-faithful']['aggregate'].get('AUPRC')}")

    json.dump(t1, open(T1, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"  [{ds}] 落盘")

print("\nUSAD/AT 补跑完成")
