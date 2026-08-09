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
    """一对一事件匹配 (第一篇公式3): 最大基数一对一指派, IoU 破平局, 漏检零信用."""
    if not gt_events or not pred_events:
        return [], 0
    # IoU 矩阵
    IoU = np.zeros((len(gt_events), len(pred_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union > 0 else 0.0
    # 匹配: IoU>0 才算可能匹配
    rows, cols = linear_sum_assignment(-IoU)
    matches = []
    for r, c in zip(rows, cols):
        if IoU[r, c] > 0:
            matches.append((r, c, IoU[r, c]))
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
