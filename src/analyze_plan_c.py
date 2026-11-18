"""方案C 终裁数据包 (任务书步骤2末, 2026-11-18)
轨1/轨2分表 + 胜负矩阵 + 配对Wilcoxon(5v5) + n=5功效声明 + SMD-28双方差 + SWaT诊断
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(__file__), "..", "results")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

t1 = json.load(open(os.path.join(R, "full_v3c_track1.json"), encoding="utf-8"))
t2 = json.load(open(os.path.join(R, "full_v3c_track2.json"), encoding="utf-8"))
s28 = json.load(open(os.path.join(R, "smd28_c.json"), encoding="utf-8"))
qs = json.load(open(os.path.join(R, "q_sensitivity.json"), encoding="utf-8"))
sw = json.load(open(os.path.join(R, "swat_diag.json"), encoding="utf-8"))

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI",
            "Exathlon-1", "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]
BASES = ["PCA", "IF", "AE", "DeepSVDD", "USAD", "MS-SVDD-style", "AT-faithful"]


def mean_of(entry, method, key="AUPRC"):
    per = entry.get(method, {}).get("per_seed", [])
    v = [p[key] for p in per if key in p and p[key] == p[key]]
    return statistics.mean(v) if v else float("nan")


def per_seed(entry, method, key="AUPRC"):
    return [p[key] for p in entry.get(method, {}).get("per_seed", []) if key in p]


print("=" * 92)
print("一、轨1 胜负矩阵 (v3 均值 vs 各基线均值; ✓=胜 ✗=负; 括号=基线均值)")
print("=" * 92)
hdr = f"{'DS':12s}{'v3':>9s}"
for b in BASES:
    hdr += f"{b[:9]:>10s}"
print(hdr)
wins = {b: 0 for b in BASES}
allwin = []
for ds in DATASETS:
    e = t1.get(ds)
    if not e:
        print(f"{ds:12s} (缺)")
        continue
    v3m = mean_of(e, "ours_v3")
    row = f"{ds:12s}{v3m:>9.3f}"
    beat_all = True
    for b in BASES:
        bm = mean_of(e, b)
        w = v3m > bm
        wins[b] += int(w)
        beat_all &= w
        row += f"{'✓' if w else '✗'}({bm:.3f})"
    allwin.append(ds if beat_all else None)
    print(row + ("  [全胜]" if beat_all else ""))
print("\nv3胜场:", ", ".join(f"{b}={wins[b]}/11" for b in BASES))
print("全胜数据集:", [d for d in allwin if d])

print()
print("=" * 92)
print("二、配对 Wilcoxon (v3 vs 基线, 5v5 同种子配对; n=5 双侧最小可达 p=0.0625 — 功效声明)")
print("=" * 92)
for ds in DATASETS:
    e = t1.get(ds)
    if not e:
        continue
    v3 = per_seed(e, "ours_v3")
    sig = []
    for b in ("PCA", "USAD", "AT-faithful"):
        bs = per_seed(e, b)
        nn = min(len(v3), len(bs))
        if nn < 2:
            sig.append(f"{b}:n<2")
            continue
        try:
            diff = [v3[i] - bs[i] for i in range(nn)]
            if all(abs(d) < 1e-12 for d in diff):
                sig.append(f"{b}:全同")
                continue
            _, p = wilcoxon(diff)
            better = "v3优" if statistics.mean(v3) > statistics.mean(bs) else "v3劣"
            sig.append(f"{b}:{'sig' if p<0.05 else 'ns'}(p={p:.3f},{better})")
        except Exception as ex:
            sig.append(f"{b}:err")
    print(f"  {ds:12s} {' | '.join(sig)}")

print()
print("=" * 92)
print("三、轨2 同预算对比 (annot 并入训练; 双类分支)")
print("=" * 92)
for ds, e in t2.items():
    row = f"  {ds:12s}"
    for m in ("ours_v3_annot", "bag_ce_only_annot", "TAIL-MIL"):
        a = e.get(m, {}).get("aggregate", {}).get("AUPRC")
        row += f"  {m}={a if a else 'FAIL/未适配'}"
    err = e.get("TAIL-MIL", {}).get("error")
    if err:
        row += f"  [TAIL-MIL error: {err[:50]}]"
    print(row)

print()
print("四、SMD-28 双方差 (实体间 vs 种子间, 分开标注)")
ok_entities = {k: v for k, v in s28.items() if "AUPRC_mean" in v}
ent_means = [v["AUPRC_mean"] for v in ok_entities.values()]
macro = statistics.mean(ent_means)
ent_sd = statistics.stdev(ent_means)
seed_sds = []
for v in ok_entities.values():
    per = v["per_seed"]
    auprcs = [p["AUPRC"] for p in per]
    if len(auprcs) > 1:
        seed_sds.append(statistics.stdev(auprcs))
macro_per_seed = [statistics.mean([v["per_seed"][i]["AUPRC"]
                                   for v in ok_entities.values()
                                   if len(v["per_seed"]) > i]) for i in range(5)]
macro_seed_sd = statistics.stdev(macro_per_seed) if len(macro_per_seed) > 1 else 0
print(f"  宏平均={macro:.4f}  实体间SD={ent_sd:.4f}  跨种子宏平均SD={macro_seed_sd:.4f}")
print(f"  单实体m-1-1={s28.get('machine-1-1', {}).get('AUPRC_mean', float('nan')):.4f}")

print()
print("五、Q 敏感性 (radius_q)")
for key, e in qs.items():
    print(f"  {key:14s} {e['aggregate'].get('AUPRC')}")

print()
print("六、SWaT 诊断")
print(f"  轨1 per-seed: {sw.get('track1_per_seed')}")
print(f"  双峰: {sw.get('bimodal')}")
if "warmup_aggregate" in sw:
    print(f"  warmup(freeze2): {sw['warmup_aggregate'].get('AUPRC')}")

print()
print("七、与旧表(four-block 泄漏协议)的不可比声明素材")
old_main = {"SMD": 0.511, "PSM": 0.463, "MSL": 0.298, "SMAP": 0.199, "SWaT": 0.520,
            "WADI": 0.025, "Exathlon-1": 0.999, "Exathlon-2": 0.970,
            "Genesis": 0.073, "Daphnet": 0.532, "CATSv2": 0.114}
print(f"  {'DS':12s}{'旧表':>8s}{'轨1新':>9s}{'Δ':>9s}")
for ds in DATASETS:
    e = t1.get(ds)
    if not e:
        continue
    v3m = mean_of(e, "ours_v3")
    print(f"  {ds:12s}{old_main.get(ds, float('nan')):>8.3f}{v3m:>9.3f}{v3m - old_main.get(ds, float('nan')):>+9.3f}")
