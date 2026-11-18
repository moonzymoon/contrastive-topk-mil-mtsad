"""AUPRC 对比图 v3 (程序化, 数据从 full_v3c_track1.json 读取 — 铁律2)
11数据集 × [Ours + 7基线], 分隔线区分 official-split / single-stream.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statistics

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = r"D:\0科研\工作1\第2篇SCI\Contrastive_TopK_MIL\paper\figures\fig_auprc_comparison.png"

t1 = json.load(open(os.path.join(R, "full_v3c_track1.json"), encoding="utf-8"))
DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI", "Exathlon-1",
            "Exathlon-2", "Genesis", "Daphnet", "CATSv2"]
LABELS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI", "Ex-1", "Ex-2",
          "Gen", "Daph", "CAT"]
METHODS = ["ours_v3", "PCA", "AE", "IF", "AT-faithful", "DeepSVDD",
           "USAD", "MS-SVDD-style"]
NAMES = {"ours_v3": "Ours", "PCA": "PCA", "AE": "AE", "IF": "IF",
         "AT-faithful": "AT", "DeepSVDD": "DeepSVDD", "USAD": "USAD",
         "MS-SVDD-style": "MS-SVDD-st."}
PALETTE = {"ours_v3": "#DDE6F3", "PCA": "#7FA4C8", "AE": "#B59BC8",
           "IF": "#D08AB6", "AT-faithful": "#8FB79B", "DeepSVDD": "#6A5A8F",
           "USAD": "#3F345B", "MS-SVDD-style": "#A33E2E"}

def mean_of(e, m):
    v = [p["AUPRC"] for p in e.get(m, {}).get("per_seed", []) if "AUPRC" in p]
    return statistics.mean(v) if v else None

data = {m: [mean_of(t1[ds], m) for ds in DATASETS] for m in METHODS}
colors = [PALETTE[m] for m in METHODS]

fig, ax = plt.subplots(figsize=(16, 5.8))
n, nm = len(DATASETS), len(METHODS)
bw = 0.10
x = np.arange(n)
for i, m in enumerate(METHODS):
    off = (i - nm / 2 + 0.5) * bw
    vals = [v if v is not None else 0 for v in data[m]]
    ax.bar(x + off, vals, bw, label=NAMES[m], color=colors[i],
           edgecolor="white", linewidth=0.4)

ax.axvline(x=5.5, color="#999", linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(2.6, 1.06, "official-split datasets (6)", ha="center",
        fontsize=10, color="#555", style="italic")
ax.text(8.5, 1.06, "single-stream TSB-AD (5)", ha="center",
        fontsize=10, color="#555", style="italic")

# Ours 全胜数据集星标 (程序化判定)
for j, ds in enumerate(DATASETS):
    v3 = data["ours_v3"][j]
    others = [data[m][j] for m in METHODS if m != "ours_v3" and data[m][j] is not None]
    if v3 is not None and others and v3 > max(others):
        ax.text(x[j], 1.02, "★", ha="center", va="bottom", fontsize=13,
                color=PALETTE["AT-faithful"])

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel("AUPRC", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title("AUPRC under the unified dual-track protocol. "
             "★ = our method best overall.", fontsize=12, pad=10)
ax.legend(loc="upper right", ncol=4, fontsize=9, frameon=True,
          edgecolor="#ccc", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
allwin = [LABELS[j] for j in range(n)
          if data["ours_v3"][j] is not None
          and all(o is None or data["ours_v3"][j] > o
                  for m in METHODS if m != "ours_v3"
                  for o in [data[m][j]])
          and sum(1 for m in METHODS if m != "ours_v3" and data[m][j] is not None) > 0]
print(f"已保存 {OUT}")
print("全胜:", allwin)
