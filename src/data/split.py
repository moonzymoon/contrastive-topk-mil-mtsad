"""防泄漏时序划分 + Bag 构造
严格沿用第一篇协议: 时序不相交 35%/15%/50%, 阈值在校准块选, 不触测试标签.
Bag 标签 = any-point 规则合成 (第一篇公式1)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class Split:
    weak_train: slice   # 0 ~ 0.35
    calib: slice        # 0.35 ~ 0.50
    test: slice         # 0.50 ~ 1.00


def chronological_split(T: int, weak_train_ratio=0.35, calib_ratio=0.15) -> Split:
    """时序不相交划分. 返回三个 slice."""
    a = int(T * weak_train_ratio)
    b = int(T * (weak_train_ratio + calib_ratio))
    return Split(slice(0, a), slice(a, b), slice(b, T))


def make_bags(X: np.ndarray, Y_point: np.ndarray, L=128, l=16, stride=64):
    """从时序滑动构造 bag.
    X: (T, D), Y_point: (T,) 点级标签
    返回 bags: list[(X_bag(n,l,D), y_bag∈{0,1})]
    bag 标签 = any-point: bag 内任一点异常 -> y_bag=1 (第一篇公式1).
    instance: 长度 l 的子段, 每 bag 含 n = (L-l)//stride + 1 个 instance.
    """
    T, D = X.shape
    bags = []
    bag_start = 0
    while bag_start + L <= T:
        seg_X = X[bag_start:bag_start + L]
        seg_Y = Y_point[bag_start:bag_start + L]
        # 切 instance
        insts = []
        ins_start = 0
        while ins_start + l <= L:
            insts.append(seg_X[ins_start:ins_start + l])
            ins_start += stride
        insts = np.stack(insts)  # (n, l, D)
        y_bag = int(seg_Y.max() > 0)  # any-point 规则
        bags.append((insts, y_bag))
        bag_start += stride
    return bags


def bags_to_array(bags):
    """bag list -> (N, n, l, D), (N,)."""
    Xb = np.stack([b[0] for b in bags])
    Yb = np.array([b[1] for b in bags], dtype=np.int64)
    return Xb, Yb


def weak_label_perturb(Yb: np.ndarray, kind: str = "none", rate=0.0, rng=None):
    """预设弱标签扰动 (第一篇8种扰动的实现接口).
    kind ∈ {none, delete_fp, inject_fp, timeshift, dilate, erode}
    阴性结果完整留存, 不美化."""
    rng = rng or np.random.default_rng(0)
    Yp = Yb.copy()
    if kind == "none" or rate == 0:
        return Yp
    if kind == "inject_fp":        # 注入假阳
        idx = rng.where(Yb == 0)[0] if hasattr(rng, "where") else np.where(Yb == 0)[0]
        n = int(len(idx) * rate)
        flip = rng.choice(idx, n, replace=False)
        Yp[flip] = 1
    elif kind == "delete_fp":      # 删除真阳(制造漏标)
        idx = np.where(Yb == 1)[0]
        n = int(len(idx) * rate)
        flip = rng.choice(idx, n, replace=False)
        Yp[flip] = 0
    # timeshift/dilate/erode 需点级操作, 留接口
    return Yp
