"""保形校准告警层 (创新点②核心)
split conformal: 在校准块(15%)算 bag 分数 -> 取分位数 q_{1-α} 为阈值 -> 迁移测试块.
全程不触测试标签 (与第一篇防泄漏协议完全兼容).
误报率保证: P(正常bag被告警) ≤ α (严格交换性下; 近似+漂移有界见论文)."""
import numpy as np


def split_conformal_threshold(calib_scores, alpha=0.05):
    """标准 split conformal: 取校准分数的 (1-α) 分位数为阈值.
    calib_scores: 校准块 bag 分数 (仅正常 bag 用于保证误报率)."""
    calib_scores = np.asarray(calib_scores, dtype=np.float64)
    if calib_scores.size == 0:
        return float("inf")
    # 取 ceil((n+1)(1-α))/n 分位, 有限样本修正
    n = calib_scores.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(calib_scores, q_level, method="higher"))


def conformal_alarm(point_scores_1d, threshold):
    """点级告警: score > threshold -> 告警点 -> 合并为事件.
    point_scores_1d: 测试块点级分数 (T_test,)
    threshold: 校准块迁移来的保形阈值 (不触测试标签)."""
    alarm = (point_scores_1d > threshold).astype(np.int64)
    return alarm


def events_from_binary(binary, min_gap=1):
    """连续告警段 -> 事件区间列表 [(start,end), ...]. end exclusive."""
    events = []
    in_evt = False
    start = 0
    for i, v in enumerate(binary):
        if v and not in_evt:
            start = i; in_evt = True
        elif not v and in_evt:
            events.append((start, i)); in_evt = False
    if in_evt:
        events.append((start, len(binary)))
    return events
