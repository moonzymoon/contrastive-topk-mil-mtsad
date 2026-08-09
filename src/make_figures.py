"""论文图表生成(真实5种子数据). Fig1损失/Fig2坍缩/Fig4数据集对比/Fig5 FAR."""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_loss_curve():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log_path = os.path.join(PROJECT_ROOT, "logs", "smd_balance.txt")
    if not os.path.exists(log_path):
        print("log不存在,跳过Fig1"); return
    epochs, totals, bags, compacts, seps = [], [], [], [], []
    for line in open(log_path, encoding="utf-8"):
        m = re.search(r"epoch (\d+)/\d+ loss=([\d.]+).*'bag': ([\d.]+).*'compact': ([\d.]+).*'sep': ([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1))); totals.append(float(m.group(2)))
            bags.append(float(m.group(3))); compacts.append(float(m.group(4))); seps.append(float(m.group(5)))
    if not epochs: print("无loss数据,跳过"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, totals, "k-", label="total", linewidth=2)
    ax.plot(epochs, bags, label="bag BCE")
    ax.plot(epochs, compacts, label="compact")
    ax.plot(epochs, seps, label="separation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("CMH-MIL training loss (SMD)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_loss_curve.png"), dpi=150)
    print("Fig.1 已生成")


def fig_collapse_diagnosis():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = list(range(1, 13))
    no_bal = [[55,74,51,44,41,42,55,62,39,33,16,9],
              [561,542,565,572,575,574,559,552,575,582,600,611],
              [4,4,4,4,4,4,6,6,6,5,4,0]]
    bal = [[131,128,129,147,153,151,126,123,139,155,155,144],
           [483,486,485,463,453,457,488,491,469,451,449,462],
           [6,6,6,10,14,12,6,6,12,14,16,14]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    cls = ["#E91E63", "#2196F3", "#4CAF50"]; mks = ["o", "s", "^"]
    for j in range(3):
        ax1.plot(epochs, no_bal[j], mks[j]+"-", color=cls[j], label=f"sphere {j+1}")
        ax2.plot(epochs, bal[j], mks[j]+"-", color=cls[j], label=f"sphere {j+1}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Sample count"); ax1.set_title("(a) Without balance regulariser (collapse)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Sample count"); ax2.set_title("(b) With balance regulariser (stable)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_collapse_diagnosis.png"), dpi=150)
    print("Fig.2 已生成")


def fig_dataset_comparison():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = ["SMD", "PSM", "MSL", "SMAP"]
    cmh_auprc = [0.444, 0.458, 0.248, 0.223]
    cmh_event = [0.410, 0.262, 0.310, 0.306]
    pca_auprc = [0.652, 0.515, 0.277, 0.137]
    pca_event = [0.286, 0.400, 0.348, 0.188]
    ae_auprc = [0.541, 0.511, 0.277, 0.135]
    ae_event = [0.152, 0.199, 0.336, 0.189]
    dmsvdd_auprc = [0.462, 0.416, 0.213, 0.188]
    dmsvdd_event = [0.373, 0.240, 0.298, 0.285]
    x = np.arange(len(datasets)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(x-1.5*w, cmh_auprc, w, label="CMH-MIL", color="#2196F3")
    ax1.bar(x-0.5*w, pca_auprc, w, label="PCA", color="#FF9800")
    ax1.bar(x+0.5*w, ae_auprc, w, label="AE", color="#9C27B0")
    ax1.bar(x+1.5*w, dmsvdd_auprc, w, label="DMSVDD", color="#4CAF50")
    ax1.set_xticks(x); ax1.set_xticklabels(datasets); ax1.set_ylabel("AUPRC")
    ax1.set_title("(a) AUPRC"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y"); ax1.set_ylim(0,0.72)
    ax2.bar(x-1.5*w, cmh_event, w, label="CMH-MIL", color="#2196F3")
    ax2.bar(x-0.5*w, pca_event, w, label="PCA", color="#FF9800")
    ax2.bar(x+0.5*w, ae_event, w, label="AE", color="#9C27B0")
    ax2.bar(x+1.5*w, dmsvdd_event, w, label="DMSVDD", color="#4CAF50")
    ax2.set_xticks(x); ax2.set_xticklabels(datasets); ax2.set_ylabel("Event-F1")
    ax2.set_title("(b) Event-F1 (one-to-one)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y"); ax2.set_ylim(0,0.45)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_dataset_comparison.png"), dpi=150)
    print("Fig.4 已生成")


def fig_far_comparison():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = ["SMD", "PSM", "MSL", "SMAP"]
    far = [0.322, 0.181, 0.027, 0.016]
    far_err = [0.062, 0.055, 0.007, 0.013]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E53935" if f > 0.05 else "#43A047" for f in far]
    ax.bar(x, far, 0.55, color=colors, yerr=far_err, capsize=5)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label=r"nominal $\alpha=0.05$")
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("Empirical false-alarm rate")
    ax.set_title("Empirical FAR vs nominal $\\alpha$")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, f in enumerate(far):
        ax.text(i, f + 0.012, f"{f:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig5_far_comparison.png"), dpi=150)
    print("Fig.5 已生成")


if __name__ == "__main__":
    for fn in [fig_loss_curve, fig_collapse_diagnosis, fig_dataset_comparison, fig_far_comparison]:
        try: fn()
        except Exception as e: print(f"{fn.__name__} 失败: {e}")
    print(f"图表目录: {FIG_DIR}")
