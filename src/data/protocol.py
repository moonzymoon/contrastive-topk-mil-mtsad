"""W1 统一协议层 (单一事实源) — IJMLC 拒稿重建 2026-11-17
==========================================
四项修复 (总管令 W1-①②③④):
① 统一切分: 四块 35/10/10/45, 主方法与全部基线同一测试区间 [0.55T, T)
② bag/instance 构造修复: bag_stride 与 instance_stride 分离, instance 全覆盖 bag
③ 评测层级统一: instance 分数→点级分数 (重叠聚合), 全部指标在时间点层级计算
④ 预处理泄漏修复: 先切分, 标准化统计只从协议 train 块拟合

纪律 (论文纪律_三条新规):
- 本模块是唯一协议入口; 任何实验脚本禁止自带切分/标准化/评测
- loaders 返回原始 X/Y (未标准化), 标准化职责在本层
"""
import numpy as np

# ===== ① 统一四块切分 (与论文声称一致) =====
TRAIN_RATIO = 0.35
VAL_RATIO = 0.10    # 35%~45%, 仅用于超参选择 (W1阶段暂不消费, 保留占位)
CALIB_RATIO = 0.10  # 45%~55%, 阈值校准
TEST_RATIO = 0.45   # 55%~100%


def four_block_split(T: int) -> dict:
    """四块时序切分. 返回各块的 (start, end) 闭开区间.
    用 round 避免浮点累加误差 (0.35+0.10 的浮点和 < 0.45 导致 int 截断偏小)."""
    a = round(T * TRAIN_RATIO)
    b = round(T * (TRAIN_RATIO + VAL_RATIO))
    c = round(T * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))
    return {"train": (0, a), "val": (a, b), "calib": (b, c), "test": (c, T)}


# ===== ④ 先切分后标准化 =====
class Scaler:
    """z-score (可选 clip), 统计只来自 fit 传入的协议 train 块."""

    def __init__(self, X_train: np.ndarray, clip_percentile=None):
        if clip_percentile is not None:
            lo, hi = np.percentile(X_train, clip_percentile, axis=0)
            self.clip = (lo, hi)
            X_train = np.clip(X_train, lo, hi)
        else:
            self.clip = None
        self.mu = X_train.mean(axis=0, keepdims=True)
        self.sigma = X_train.std(axis=0, keepdims=True) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.clip is not None:
            X = np.clip(X, *self.clip)
        return ((X - self.mu) / self.sigma).astype(np.float32)


def make_protocol_pipeline(X: np.ndarray, Y: np.ndarray):
    """W1 标准管线: 切分→train块拟合标准化→全流变换.
    返回 dict(Xs=标准化后全流, split=四块索引, scaler)."""
    sp = four_block_split(len(X))
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    return {"Xs": scaler.transform(X), "split": sp, "scaler": scaler}


# ===== ② bag/instance 构造 (修复版) =====
def make_bags_v2(X, Y_point, L=128, l=16, bag_stride=64, instance_stride=None):
    """bag 滑窗 + instance 全覆盖构造 (修复 R3: 旧版 instance 只覆盖 32/128 点).

    instance_stride 默认 = l → instance 起点 0, l, 2l, ..., L-l, 全覆盖 bag.
    bag 标签 = any-point (bag 内任一点异常). instance 全覆盖后标签与
    instance 观测一致 (异常 bag 必含异常 instance).

    返回:
        insts:   (N, n, l, D) float32
        y_bag:   (N,) int8
        starts:  (N,) bag 在原流中的起点
        inst_rel: (n,) instance 在 bag 内的相对起点 (所有 bag 共享)
    """
    if instance_stride is None:
        instance_stride = l
    assert instance_stride <= l, "instance_stride 必须 <= l 才能全覆盖"
    n = (L - l) // instance_stride + 1
    inst_rel = np.arange(n) * instance_stride  # (n,)
    T, D = X.shape
    bags, labels, starts = [], [], []
    bs = 0
    while bs + L <= T:
        seg = X[bs:bs + L]
        insts = np.stack([seg[s:s + l] for s in inst_rel])  # (n,l,D)
        y = int(Y_point[bs:bs + L].max() > 0)
        bags.append(insts); labels.append(y); starts.append(bs)
        bs += bag_stride
    if not bags:
        raise ValueError(f"流长 T={T} 不足以构造一个 bag (L={L})")
    return (np.stack(bags).astype(np.float32),
            np.array(labels, dtype=np.int8),
            np.array(starts, dtype=np.int64),
            inst_rel)


# ===== ③ instance 分数 → 点级分数 =====
def point_scores(T, bag_starts, inst_rel, inst_scores, l=16, agg="max"):
    """instance 分数映射回原始时间轴, 重叠处聚合 (R2/R3 修复).

    inst_scores: (N, n) 每 bag 每 instance 的分数.
    返回 (s_point, covered):
        s_point: (T,) float32, 未覆盖点为 NaN
        covered: (T,) bool, 被至少一个 instance 覆盖的点
    """
    assert agg in ("max", "mean")
    s = np.full(T, -np.inf, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int32)
    acc = np.zeros(T, dtype=np.float64)
    N, n = inst_scores.shape
    for i in range(N):
        b0 = bag_starts[i]
        for j in range(n):
            a = b0 + inst_rel[j]
            b = a + l
            seg = inst_scores[i, j]
            if agg == "max":
                s[a:b] = np.maximum(s[a:b], seg)
            else:
                acc[a:b] += seg; cnt[a:b] += 1
    if agg == "mean":
        s = np.where(cnt > 0, acc / np.maximum(cnt, 1), -np.inf)
    covered = np.isfinite(s) & (s > -np.inf)
    s_point = np.where(covered, s, np.nan).astype(np.float32)
    return s_point, covered


# ===== ③ 点级评测 =====
def threshold_from_calib(s_calib, y_calib_point, alpha=0.05):
    """校准块正常点分数的 (1-alpha) 分位作阈值 (conformal 思想, 点级)."""
    normal = s_calib[y_calib_point == 0]
    normal = normal[~np.isnan(normal)]
    if len(normal) == 0:
        raise ValueError("校准块无正常点")
    return float(np.quantile(normal, 1 - alpha))


def evaluate_point(Y_point, s_point, thr, covered=None):
    """点级指标: AUPRC/AUROC/PointF1/EventF1(一对一)/FAR.
    只在 covered 点上评测; 未覆盖点排除 (覆盖率一并返回)."""
    from evaluation.metrics import evaluate_all
    from evaluation.conformal import events_from_binary
    if covered is None:
        covered = ~np.isnan(s_point)
    y = Y_point[covered].astype(int)
    s = s_point[covered]
    pred = (s > thr).astype(int)
    m = evaluate_all(y, s, thr,
                     events_from_binary(y), events_from_binary(pred))
    m["FAR_point"] = float(pred[y == 0].mean()) if (y == 0).sum() > 0 else float("nan")
    m["coverage"] = float(covered.mean())
    m["eval_points"] = int(covered.sum())
    return m


def run_point_evaluation(X, Y_point, score_fn, alpha=0.05, L=128, l=16,
                         bag_stride=64, instance_stride=None, agg="max"):
    """W1 一站式点级评测管线.

    score_fn: (N,n,l,D)->(N,n) instance 分数函数 (由调用方模型提供)
    流程: 切分→标准化→各块构造bag→instance分数→点级映射→calib阈值→点级指标
    """
    pipe = make_protocol_pipeline(X, Y_point)
    Xs, sp = pipe["Xs"], pipe["split"]

    out = {}
    s_point_all = np.full(len(Xs), np.nan, dtype=np.float32)
    for name in ("calib", "test"):
        s_blk, e_blk = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(
            Xs[s_blk:e_blk], Y_point[s_blk:e_blk], L, l, bag_stride, instance_stride)
        scores = score_fn(insts)  # (N, n)
        s_pt, cov = point_scores(e_blk - s_blk, starts, rel, scores, l, agg)
        # 映射回全流坐标
        s_point_all[s_blk:e_blk] = s_pt
        out[f"{name}_bags"] = int(len(y_bag))
        out[f"{name}_anom_bags"] = int(y_bag.sum())

    thr = threshold_from_calib(s_point_all[sp["calib"][0]:sp["calib"][1]],
                               Y_point[sp["calib"][0]:sp["calib"][1]], alpha)
    covered = ~np.isnan(s_point_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y_point[sp["test"][0]:sp["test"][1]],
                       s_point_all[sp["test"][0]:sp["test"][1]], thr, covered)
    m["threshold"] = thr
    m["train_anom_bags"] = None  # 由调用方填充 (弱监督设定记录)
    return m


# ===== 冒烟单测 (博1纪律: 新组件先过单测再接入) =====
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rng = np.random.default_rng(0)
    # 合成流: T=2000, D=4, 点 900-950 与 1600-1650 异常
    T, D = 2000, 4
    X = rng.normal(0, 1, (T, D)).astype(np.float32)
    Y = np.zeros(T, dtype=np.int8)
    Y[900:950] = 1; Y[1600:1650] = 1
    X[900:950] += 5.0; X[1600:1650] += 5.0

    sp = four_block_split(T)
    # 性质验证: 顺序单调、互不重叠、覆盖全流、比例误差<2点
    assert sp["train"][1] == sp["val"][0] and sp["val"][1] == sp["calib"][0] \
        and sp["calib"][1] == sp["test"][0] and sp["test"][1] == T
    assert abs(sp["train"][1] - 700) <= 2 and abs(sp["test"][0] - 1100) <= 2
    print("[1/5] 四块切分 OK:", sp)

    pipe = make_protocol_pipeline(X, Y)
    tr = pipe["Xs"][sp["train"][0]:sp["train"][1]]
    assert abs(tr.mean()) < 0.05 and abs(tr.std() - 1) < 0.05, "train块应标准化为~N(0,1)"
    # 泄漏检查: scaler 统计不得依赖 700 之后的行
    X2 = X.copy(); X2[1500:] += 100.0
    sc2 = Scaler(X2[sp["train"][0]:sp["train"][1]])
    assert np.allclose(sc2.mu, pipe["scaler"].mu), "标准化统计必须与未来数据无关"
    print("[2/5] 先切分后标准化 OK (未来数据不改统计)")

    insts, y_bag, starts, rel = make_bags_v2(X[:1000], Y[:1000], L=128, l=16, bag_stride=64)
    n_cov = len(rel) * 16
    # 注: 旧论文写 n=7 是笔误 (7*16=112<128 无法覆盖); 正确全覆盖 n=8 (起点0..112), W4 更正论文
    assert insts.shape[1] == 8, f"n应为8(全覆盖), 实际{insts.shape[1]}"
    assert n_cov == 128, "instance 必须全覆盖 bag"                     # R3 修复
    # 异常 bag 必含异常 instance (any-point 一致性)
    ai = [i for i, yb in enumerate(y_bag) if yb == 1]
    for i in ai:
        bs = starts[i]
        assert any(Y[bs + r:bs + r + 16].max() > 0 for r in rel), "异常bag必须含异常instance"
    print(f"[3/5] bag构造 OK: n=8, 全覆盖128点, 异常bag {len(ai)} 个全部含异常instance")

    # 点级映射: 分数=instance内均值 → 异常点分数应高
    scores = insts.mean(axis=(2, 3))  # (N, n)
    s_pt, cov = point_scores(1000, starts, rel, scores, l=16, agg="max")
    anom_pt = s_pt[900:950]; normal_pt = s_pt[400:600]
    assert cov[900:950].all(), "异常区间必须被覆盖"
    assert anom_pt.mean() > normal_pt.mean() + 1, "异常点分数应显著高于正常"
    print(f"[4/5] 点级映射 OK: 异常均分{anom_pt.mean():.2f} >> 正常均分{normal_pt.mean():.2f}")

    # 端到端: 完美分数函数 → 点级指标
    def perfect_score(insts_batch):
        return insts_batch.mean(axis=(2, 3))
    m = run_point_evaluation(X, Y, perfect_score)
    assert m["AUPRC"] > 0.9, f"完美分数AUPRC应>0.9, 实际{m['AUPRC']}"
    print(f"[5/5] 端到端点级评测 OK: AUPRC={m['AUPRC']:.3f} PointF1={m['PointF1']:.3f} "
          f"EventF1={m['EventF1_one2one']:.3f} FAR={m['FAR_point']:.3f} 覆盖率={m['coverage']:.3f}")
    print("\n全部冒烟测试通过 — W1 协议层可接入实验脚本")
