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
    """一对一事件匹配: 最大基数优先, 同基数下最大总IoU, 漏检零信用.
    R4-P0-3 (2026-11-18 三外审): 旧贪心不保证最大基数 (高IoU对可阻塞次高IoU对
    的互补配对). 改用 scipy maximum_bipartite_matching 求真最大基数,
    再在最大基数解中选总IoU最大的 (通过 Hungarian on -IoU restricted to
    maximum-cardinality subgraph)."""
    if not gt_events or not pred_events:
        return [], 0
    IoU = np.zeros((len(gt_events), len(pred_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union > 0 else 0.0
    # 二部图: IoU>0 的边
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    adj = (IoU > 0).astype(np.int32)
    # scipy 需要 (n_pred, n_gt) 方向
    match = maximum_bipartite_matching(csr_matrix(adj.T), perm_type='column')
    # match[i] = j 表示 pred i → gt j; 转换为 (gt, pred) 对
    base_matches = []
    for pred_idx, gt_idx in enumerate(match):
        if gt_idx >= 0:  # -1 = unmatched
            base_matches.append((gt_idx, pred_idx, IoU[gt_idx, pred_idx]))
    # 最大基数已保证; 在此基础上用 Hungarian 选总IoU最大的最大基数匹配
    if len(base_matches) > 0:
        # 构建限制图: 仅保留有正IoU的行/列, Hungarian 最大化总IoU但限定基数
        from scipy.optimize import linear_sum_assignment
        # 大IoU矩阵(求最小化): 对IoU>0的子矩阵做 Hungarian (-IoU), 无边处设大值
        n_g, n_p = IoU.shape
        cost = np.where(IoU > 0, -IoU, 1e6)
        rows, cols = linear_sum_assignment(cost)
        hung_matches = [(r, c, IoU[r, c]) for r, c in zip(rows, cols)
                        if IoU[r, c] > 0]
        # Hungarian 可能给出更大基数; 取两者中基数大的
        if len(hung_matches) >= len(base_matches):
            matches = sorted(hung_matches, key=lambda m: m[0])
        else:
            matches = sorted(base_matches, key=lambda m: m[0])
    else:
        matches = []
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
