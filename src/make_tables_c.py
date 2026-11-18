"""方案C 表格程序化生成 (任务书步骤4-2, 铁律2: 数字零手填)
从 full_v3c_track1.json / full_v3c_track2.json / q_sensitivity.json 生成:
  tables_c.tex  (主表 + 轨2表 + 敏感性表, 论文 \\input)
  tables_audit.json (图-表-JSON三方对账源)
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_TEX_DIR = r"D:\0科研\工作1\第2篇SCI\Contrastive_TopK_MIL\paper"
OUT_AUDIT = os.path.join(R, "tables_audit.json")

t1 = json.load(open(os.path.join(R, "full_v3c_track1.json"), encoding="utf-8"))
t2 = json.load(open(os.path.join(R, "full_v3c_track2.json"), encoding="utf-8"))
qs = json.load(open(os.path.join(R, "q_sensitivity.json"), encoding="utf-8"))
s28 = json.load(open(os.path.join(R, "smd28_c.json"), encoding="utf-8"))

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "WADI",
            "Exathlon-1", "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]  # SWaT excluded (R4)
DS_LABEL = {"SMD": "SMD (m-1-1)", "PSM": "PSM", "MSL": "MSL", "SMAP": "SMAP",
            "WADI": "WADI", "Exathlon-1": "Exathlon-1",
            "Exathlon-2": "Exathlon-2", "Genesis": "Genesis",
            "Daphnet": "Daphnet", "CATSv2": "CATSv2"}
BASES = ["PCA", "IF", "AE", "DeepSVDD", "USAD", "MS-SVDD-style", "AT-faithful"]
BASE_TEX = {"PCA": "PCA", "IF": "IF", "AE": "AE", "DeepSVDD": "DeepSVDD",
            "USAD": "USAD", "MS-SVDD-style": "MS-SVDD-st.", "AT-faithful": "AT"}
audit = {"main_table": {}, "track2": {}, "q": {}}


def meanstd(entry, method, key="AUPRC"):
    v = [p[key] for p in entry.get(method, {}).get("per_seed", [])
         if key in p and p[key] == p[key]]
    if not v:
        return None
    m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    return m, s, len(v)


def fmt(ms, bold=False, with_std=True):
    if ms is None:
        return "---"
    m, s, _ = ms
    cell = f"{m:.3f}" + (f"$\\pm${s:.3f}" if with_std else "")
    return (f"\\textbf{{{cell}}}" if bold else cell)


# ===== 主表 (轨1) =====
lines = [r"\begin{table}[H]\centering",
         r"\caption{AUPRC on eleven benchmarks under the unified protocol "
         r"(four-block official-train split for SMD/PSM/MSL/SMAP/SWaT/WADI; "
         r"single-stream for the TSB-AD datasets; point-level metrics; 5 seeds for "
         r"all methods). Bold = best per dataset. ``---'' = not applicable.}",
         r"\label{tab:main}", r"\scriptsize",
         r"\begin{tabular}{l" + "c" * (1 + len(BASES)) + "}", r"\toprule",
         "Dataset & \\textbf{Ours} & " + " & ".join(BASE_TEX[b] for b in BASES) + r" \\",
         r"\midrule"]
main_audit = {}
for ds in DATASETS:
    e = t1[ds]
    row = [DS_LABEL[ds]]
    ours = meanstd(e, "ours_v3")
    scores = {b: meanstd(e, b) for b in BASES}
    allv = [ours] + [scores[b] for b in BASES]
    best = max(v[0] for v in allv if v)
    row.append(fmt(ours, bold=(ours and ours[0] == best)))
    for b in BASES:
        v = scores[b]
        row.append(fmt(v, bold=(v and v[0] == best)))
        main_audit.setdefault(b, {})[ds] = round(v[0], 4) if v else None
    main_audit.setdefault("ours_v3", {})[ds] = round(ours[0], 4) if ours else None
    lines.append(" & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
main_tex = "\n".join(lines)

# ===== 轨2表 (同预算) =====
lines2 = [r"\begin{table}[H]\centering",
          r"\caption{Track-2 equal-label-budget comparison (weak-annotation block "
          r"consumed by all three methods): AUPRC, 5-seed mean$\pm$std. "
          r"bag-CE = architecture identical to ours with the contrastive and TopK "
          r"objectives disabled (isolates the contrastive contribution, "
          r"Reviewer-2 question). TAIL-MIL uses its official model.}",
          r"\label{tab:track2}", r"\small",
          r"\begin{tabular}{lccc}", r"\toprule",
          r"Dataset & Ours (annot) & bag-CE only & TAIL-MIL \\",
          r"\midrule"]
track2_audit = {}
for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:  # SWaT excluded
    e = t2[ds]
    cells = []
    for m in ("ours_v3_annot", "bag_ce_only_annot", "TAIL-MIL"):
        v = meanstd(e, m)
        if v is None and "error" in e.get(m, {}):
            cells.append(r"\textit{n/a}")
            track2_audit.setdefault(m, {})[ds] = None
            continue
        cells.append(fmt(v, with_std=True))
        track2_audit.setdefault(m, {})[ds] = round(v[0], 4)
    lines2.append(f"{DS_LABEL[ds]} & " + " & ".join(cells) + r" \\")
lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
track2_tex = "\n".join(lines2)

# ===== 敏感性表 =====
linesq = [r"\begin{table}[H]\centering",
          r"\caption{Quantile level $q$ sensitivity (radius EMA level; AUPRC, 3 seeds).}",
          r"\label{tab:qsens}", r"\small",
          r"\begin{tabular}{lccc}", r"\toprule",
          r"Dataset & $q{=}0.90$ & $q{=}0.95$ & $q{=}0.99$ \\",
          r"\midrule"]
for ds in ("MSL", "SMD"):
    cells = []
    for q in ("0.9", "0.95", "0.99"):
        v = meanstd(qs, f"{ds}_q{q}")
        cells.append(fmt(v, with_std=False))
        audit["q"].setdefault(ds, {})[q] = round(v[0], 4) if v else None
    linesq.append(f"{ds} & " + " & ".join(cells) + r" \\")
linesq += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
q_tex = "\n".join(linesq)

# ===== SMD-28 数字 (供正文引用) =====
ok = [v["AUPRC_mean"] for v in s28.values() if "AUPRC_mean" in v]
m28 = statistics.mean(ok); s_ent = statistics.stdev(ok)
per_seed_macro = []
for i in range(5):
    per_seed_macro.append(statistics.mean(
        [v["per_seed"][i]["AUPRC"] for v in s28.values()
         if "AUPRC_mean" in v and len(v["per_seed"]) > i]))
s_seed = statistics.stdev(per_seed_macro)
audit["smd28"] = {"macro_mean": round(m28, 4), "entity_sd": round(s_ent, 4),
                  "seed_sd": round(s_seed, 4), "m11": s28["machine-1-1"]["AUPRC_mean"]}


# ===== EF1/FAR 表 (R4-P0-4: 程序化替代手填) =====
ef1_lines = [r"\begin{table}[H]\centering",
             r"\caption{Event-F1 (one-to-one, maximum-cardinality matching) and empirical FAR (5-seed mean$\\pm$std). Auto-generated from full_v3c_track1.json.}",
             r"\label{tab:ef1}", r"\small",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Dataset & Event-F1 & FAR (empirical) \\", r"\midrule"]
ef1_audit = {}
for ds in DATASETS:
    e = t1.get(ds)
    if not e:
        continue
    cells = []
    for key in ("EventF1_one2one", "FAR_point"):
        per = [p[key] for p in e["ours_v3"]["per_seed"] if key in p and p[key] == p[key]]
        if per:
            m = statistics.mean(per)
            s = statistics.stdev(per) if len(per) > 1 else 0
            cells.append(f"{m:.3f}$\\pm${s:.3f}")
            ef1_audit.setdefault(key, {})[ds] = round(m, 4)
        else:
            cells.append("---")
            ef1_audit.setdefault(key, {})[ds] = None
    ef1_lines.append(f"{DS_LABEL[ds]} & " + " & ".join(cells) + r" \\")
ef1_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
ef1_tex = "\n".join(ef1_lines)
audit["ef1"] = ef1_audit

HDR = ("% AUTO-GENERATED by make_tables_c.py - DO NOT hand-edit numbers.\n"
       "% Source: results/full_v3c_track1.json / full_v3c_track2.json / q_sensitivity.json\n")
for fname, body in (("tables_main_c.tex", main_tex),
                    ("tables_track2_c.tex", track2_tex),
                    ("tables_q_c.tex", q_tex),
                    ("tables_ef1_c.tex", ef1_tex)):
    with open(os.path.join(OUT_TEX_DIR, fname), "w", encoding="utf-8") as f:
        f.write(HDR + body + "\n")
print("三表已生成: tables_main_c / tables_track2_c / tables_q_c")

audit["main_table"] = main_audit
audit["track2"] = track2_audit
json.dump(audit, open(OUT_AUDIT, "w", encoding="utf-8"), indent=2)
print("tables_audit.json 写入")
print(f"SMD-28: macro={m28:.4f}±entity{s_ent:.4f}, seed-SD={s_seed:.4f}, m-1-1={audit['smd28']['m11']}")
