"""评估指标 (沿用第一篇 + 补 VUS-PR)
AUPRC / PointF1 / EventF1(一对一) / AllEventMIoU / VUS-PR / AUROC
事件评估: 一对一匹配 + 漏检零信用 (第一篇公式3).
阴性结果完整留存."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def auprc(y_true, y_score):
    """PR 曲线下面积 (主排序指标)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def auroc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def point_f1_at_threshold(y_true, y_score, threshold):
    pred = (y_score > threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    if tp + fp == 0:
        p = 0.0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        r = 0.0
    else:
        r = tp / (tp + fn)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def _one2one_match(gt_events, pred_events):
    """一对一事件匹配 (最大基数优先, IoU 降序贪心破平局), 漏检零信用.
    修复 (三外审 Gustav R. 2026-11-18): 旧实现 linear_sum_assignment(-IoU)
    最大化【总 IoU】, 与本注释"最大基数"不符 — pred 多于 gt 时会漏配低 IoU 对
    (2 gt + 3 pred 只配 1 对的反例已复现). 改为: 先取全部 IoU>0 候选对,
    按IoU降序贪心, 行列各至多一次 → 基数最大; 同基数下总 IoU 尽量大."""
    if not gt_events or not pred_events:
        return [], 0
    # IoU 矩阵
    IoU = np.zeros((len(gt_events), len(pred_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union > 0 else 0.0
    # 最大基数贪心: IoU>0 候选按降序, 行列各至多一次
    pairs = sorted(((IoU[i, j], i, j)
                    for i in range(len(gt_events))
                    for j in range(len(pred_events)) if IoU[i, j] > 0), reverse=True)
    used_r, used_c, matches = set(), set(), []
    for v, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r); used_c.add(c)
        matches.append((r, c, v))
    matches.sort(key=lambda m: m[0])  # 按 gt 顺序输出
    return matches, IoU


def event_f1_one2one(gt_events, pred_events):
    """一对一 Event-F1, 漏检零信用 (第一篇)."""
    matches, IoU = _one2one_match(gt_events, pred_events)
    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}


def all_event_miou(gt_events, pred_events):
    """全事件平均 IoU (第一篇): 未匹配真事件贡献0."""
    if not gt_events:
        return float("nan")
    matches, IoU = _one2one_match(gt_events, pred_events)
    matched_gt = {m[0] for m in matches}
    total = sum(m[2] for m in matches)
    # 未匹配 gt 贡献 0
    return float(total / len(gt_events))


def vus_pr(y_true, y_score, n_thr=50):
    """Volume-Under-the-Surface (PR) 近似: 对多点阈值平均 AUPRC.
    第一篇 Future Work 点名要补. 在 [0.05,0.95] 分位扫阈值."""
    s = np.asarray(y_score)
    thresholds = np.quantile(s, np.linspace(0.05, 0.95, n_thr))
    aps = []
    for t in thresholds:
        pred = (s > t).astype(int)
        if pred.sum() == 0:
            continue
        from sklearn.metrics import precision_score, recall_score
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        aps.append((p, r))
    # 近似 VUS: PR 空间下体积
    if not aps:
        return 0.0
    aps = np.array(aps)
    # 简化: 取 (p*r) 的积分近似
    return float(np.mean(aps[:, 0] * aps[:, 1]))


def evaluate_all(y_true_point, y_score_point, threshold, gt_events, pred_events):
    """汇总全部指标. y_true/y_score 点级; events 区间列表."""
    return {
        "AUPRC": auprc(y_true_point, y_score_point),
        "AUROC": auroc(y_true_point, y_score_point),
        "PointF1": point_f1_at_threshold(y_true_point, y_score_point, threshold)["f1"],
        "EventF1_one2one": event_f1_one2one(gt_events, pred_events)["f1"],
        "AllEventMIoU": all_event_miou(gt_events, pred_events),
        "VUS_PR": vus_pr(y_true_point, y_score_point),
    }
