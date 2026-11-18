"""方案C 全量重跑主脚本 (任务书步骤2, 2026-11-18)
==========================================================
轨1: 11数据集 × 8方法 × 5种子 (统一; 经典基线也5种子 — 回应种子声明)
     SMD/PSM/MSL/SMAP/SWaT/WADI → official_split (官方train内 70/15/15)
     TSB5集 (Exathlon-1/2, Genesis, Daphnet, CATSv2) → single-stream 35/10/10/45
轨2: 6数据集 × [v3-annot (双类InfoNCE) + v3-bagCE-only (R2-4对照)] × 5种子
     (TAIL-MIL 同预算适配由 run_tailmil_c.py 单独执行 — 失败则如实入JSON)
SMD-28: 修后v3 × 28实体 × 5种子 (轨1)
敏感性: radius_q ∈ {0.90,0.95,0.99} × {MSL,SMD} × 3种子 (轨1)
SWaT诊断: 轨1五种子若双峰 → freeze_epochs=2 warmup 对照 5种子
断点续跑: 每数据集落盘. 输出 results/full_v3c_track1.json / track2.json /
smd28_c.json / q_sensitivity.json / swat_diag.json
"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from run_v3 import run_v3, load
from baselines_v3 import BASELINES_V3
from baseline_at_faithful import baseline_at_faithful

SEEDS = [7, 42, 123, 202, 999]
SEEDS_Q = [7, 42, 123]           # 敏感性用
R = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS_OFFICIAL = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]     # 轨1 official
DATASETS_TSB = ["Exathlon-1", "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]  # 轨1 single-stream
DATASETS_ALL = DATASETS_OFFICIAL + DATASETS_TSB
TRACK2_SETS = DATASETS_OFFICIAL                                        # 轨2 (annot)
CLASSIC = ["PCA", "IF"]          # 确定性 (5种子同值, 仍按5种子跑以满足统一口径)
NEURAL = ["AE", "DeepSVDD", "USAD", "MS-SVDD-style"]


def agg(per):
    a = {}
    for k in ("AUPRC", "AUROC", "PointF1", "EventF1_one2one", "FAR_point"):
        v = [p[k] for p in per if k in p and p[k] == p[k]]
        if v:
            a[k] = (f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}"
                    if len(v) > 1 else f"{v[0]:.4f}")
    return a


def save(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)


def load_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ============ 轨1 ============
    out1 = os.path.join(R, "full_v3c_track1.json")
    t1 = load_json(out1)
    for ds in DATASETS_ALL:
        if ds in t1:
            print(f"[T1 {ds}] 已有, 跳过")
            continue
        X, Y, n_off = load(ds)
        split_mode = "plan-c" if n_off is not None else "four-block"
        print(f"\n=== [T1 {ds}] T={len(Y)} split={split_mode} n_off={n_off} ===")
        entry = {"T": len(Y), "n_official_train": n_off, "split_mode": split_mode}
        # v3 主方法 5种子
        per = []
        for sd in SEEDS:
            m = run_v3(X, Y, sd, split_mode=split_mode, n_official_train=n_off)
            m["seed"] = sd
            per.append(m)
            print(f"  v3 s{sd}: AUPRC={m['AUPRC']:.4f}")
        entry["ours_v3"] = {"per_seed": per, "aggregate": agg(per)}
        # 7基线 5种子 (统一口径)
        for name, fn in BASELINES_V3.items():
            per_b = []
            for sd in SEEDS:
                try:
                    mb = fn(X, Y, seed=sd)
                    mb["seed"] = sd
                    per_b.append(mb)
                except Exception as e:
                    print(f"  {name} s{sd} FAIL: {str(e)[:60]}")
            if per_b:
                entry[name] = {"per_seed": per_b, "aggregate": agg(per_b)}
            print(f"  {name}: {entry.get(name, {}).get('aggregate', {}).get('AUPRC', 'FAIL')}")
        # AT-faithful 5种子
        per_at = []
        for sd in SEEDS:
            try:
                ma = baseline_at_faithful(X, Y, seed=sd)
                ma["seed"] = sd
                per_at.append(ma)
            except Exception as e:
                print(f"  AT s{sd} FAIL: {str(e)[:60]}")
        if per_at:
            entry["AT-faithful"] = {"per_seed": per_at, "aggregate": agg(per_at)}
        print(f"  AT-faithful: {entry.get('AT-faithful', {}).get('aggregate', {}).get('AUPRC', 'FAIL')}")
        t1[ds] = entry
        save(out1, t1)
        print(f"[T1 {ds}] 落盘 ✓")

    # ============ 轨2 ============
    out2 = os.path.join(R, "full_v3c_track2.json")
    t2 = load_json(out2)
    for ds in TRACK2_SETS:
        if ds in t2:
            print(f"[T2 {ds}] 已有, 跳过")
            continue
        X, Y, n_off = load(ds)
        print(f"\n=== [T2 {ds}] annot 双类 + bag-CE 对照, 5种子 ===")
        entry = {"T": len(Y), "n_official_train": n_off, "split_mode": "plan-c-annot"}
        for tag, kw in [("ours_v3_annot", dict(split_mode="plan-c-annot",
                                               n_official_train=n_off)),
                        ("bag_ce_only_annot", dict(split_mode="plan-c-annot",
                                                   n_official_train=n_off,
                                                   bag_ce_only=True))]:
            per = []
            for sd in SEEDS:
                m = run_v3(X, Y, sd, **kw)
                m["seed"] = sd
                per.append(m)
                print(f"  {tag} s{sd}: AUPRC={m['AUPRC']:.4f} EF1={m['EventF1_one2one']:.4f}")
            entry[tag] = {"per_seed": per, "aggregate": agg(per)}
        # TAIL-MIL 同预算: 适配失败如实记录
        try:
            from run_tailmil_c import tailmil_plan_c
            per_t = []
            for sd in SEEDS:
                mt = tailmil_plan_c(X, Y, n_off, seed=sd)
                mt["seed"] = sd
                per_t.append(mt)
                print(f"  TAIL-MIL s{sd}: AUPRC={mt['AUPRC']:.4f}")
            entry["TAIL-MIL"] = {"per_seed": per_t, "aggregate": agg(per_t)}
        except Exception as e:
            entry["TAIL-MIL"] = {"error": f"adaptation failed: {str(e)[:200]}",
                                 "note": "protocol finding, reported as-is per taskbook"}
            print(f"  TAIL-MIL 适配失败(如实记录): {str(e)[:120]}")
        t2[ds] = entry
        save(out2, t2)
        print(f"[T2 {ds}] 落盘 ✓")

    # ============ SMD-28 (轨1) ============
    out28 = os.path.join(R, "smd28_c.json")
    s28 = load_json(out28)
    from data.loaders import load_smd
    entities = ([f"machine-1-{j}" for j in range(1, 9)]
                + [f"machine-2-{j}" for j in range(1, 10)]
                + [f"machine-3-{j}" for j in range(1, 12)])
    for ent in entities:
        if ent in s28:
            continue
        try:
            X, Y, n_off = load_smd(ent)
            per = [run_v3(X, Y, sd, split_mode="plan-c", n_official_train=n_off)
                   for sd in SEEDS]
            auprcs = [p["AUPRC"] for p in per]
            s28[ent] = {"per_seed": per,
                        "AUPRC_mean": statistics.mean(auprcs),
                        "AUPRC_std": statistics.stdev(auprcs) if len(auprcs) > 1 else 0.0}
            print(f"[SMD28 {ent}] {s28[ent]['AUPRC_mean']:.4f}±{s28[ent]['AUPRC_std']:.4f}")
        except Exception as e:
            s28[ent] = {"error": str(e)[:150]}
            print(f"[SMD28 {ent}] FAIL {str(e)[:60]}")
        save(out28, s28)

    # ============ radius_q 敏感性 ============
    outq = os.path.join(R, "q_sensitivity.json")
    qs = load_json(outq)
    for ds in ("MSL", "SMD"):
        for q in (0.90, 0.95, 0.99):
            key = f"{ds}_q{q}"
            if key in qs:
                continue
            X, Y, n_off = load(ds)
            per = []
            for sd in SEEDS_Q:
                m = run_v3(X, Y, sd, split_mode="plan-c", n_official_train=n_off,
                           radius_q=q)
                per.append(m)
            qs[key] = {"per_seed": per, "aggregate": agg(per)}
            print(f"[Q {key}] {qs[key]['aggregate'].get('AUPRC')}")
            save(outq, qs)

    # ============ SWaT 诊断 (双峰→warmup对照) ============
    outw = os.path.join(R, "swat_diag.json")
    sw = load_json(outw)
    e = t1.get("SWaT", {}).get("ours_v3", {}).get("per_seed", [])
    if e:
        auprcs = [p["AUPRC"] for p in e]
        collapsed = [a for a in auprcs if a < 0.4]
        sw["track1_per_seed"] = auprcs
        sw["bimodal"] = len(collapsed) >= 2
        sw["collapsed_seeds"] = [p.get("seed") for p in e if p["AUPRC"] < 0.4]
        if sw["bimodal"]:
            X, Y, n_off = load("SWaT")
            per_w = []
            for sd in SEEDS:
                m = run_v3(X, Y, sd, split_mode="plan-c", n_official_train=n_off,
                           freeze_epochs=2)
                m["seed"] = sd
                per_w.append(m)
                print(f"[SWaT warmup s{sd}] AUPRC={m['AUPRC']:.4f} "
                      f"radii={[round(r, 3) for r in m['final_radii']]}")
            sw["warmup_freeze2_per_seed"] = per_w
            sw["warmup_aggregate"] = agg(per_w)
        save(outw, sw)
        print(f"[SWaT diag] bimodal={sw['bimodal']}, warmup done")

    print("\n" + "=" * 60)
    print("方案C 全量重跑完成: track1 / track2 / smd28_c / q_sensitivity / swat_diag")


if __name__ == "__main__":
    main()
