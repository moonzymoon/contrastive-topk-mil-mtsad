"""结果汇总: 跨种子聚合 + Friedman/Wilcoxon 统计检验
读取 results/*.json, 输出论文用表格(mean±std) + 统计检验."""
import os, json, glob, sys
import numpy as np
from scipy import stats

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
METRICS = ["AUPRC", "AUROC", "PointF1", "EventF1_one2one", "AllEventMIoU", "VUS_PR"]


def load_all():
    """按 dataset 聚合所有 seed 结果."""
    results = {}
    for f in glob.glob(os.path.join(RESULT_DIR, "*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            if "metrics" not in d:
                continue
            ds = d.get("dataset", "?")
            results.setdefault(ds, {}).setdefault("seeds", []).append(d["metrics"])
        except Exception:
            continue
    return results


def summarize(results):
    """输出 mean±std 表格."""
    print("=" * 80)
    print(f"{'Dataset':<12} | " + " | ".join(f"{m:<20}" for m in METRICS))
    print("-" * 80)
    for ds in sorted(results):
        seeds = results[ds]["seeds"]
        row = []
        for m in METRICS:
            vals = [s[m] for s in seeds if m in s and not (isinstance(s[m], float) and np.isnan(s[m]))]
            if vals:
                row.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}")
            else:
                row.append("  -  ")
        print(f"{ds:<12} | " + " | ".join(f"{r:<20}" for r in row))
    print("=" * 80)


def friedman_test(results, metric="AUPRC"):
    """Friedman 检验 (数据集为独立单元). 需多方法才有意义, 此处单方法留接口."""
    print(f"\n[Friedman 检验 - {metric}]: (需多方法对比, 当前单方法)")


if __name__ == "__main__":
    r = load_all()
    if not r:
        print("暂无结果文件"); sys.exit(0)
    summarize(r)
    for ds in sorted(r):
        seeds = r[ds]["seeds"]
        print(f"  {ds}: {len(seeds)} seeds")
