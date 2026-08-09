"""TSB-AD-M 数据集加载器.
统一加载 TSB-AD-M benchmark 的多变量子数据集.
每个 CSV: 前N-1列为特征, 最后一列 'Label' 为 0/1 异常标签.
"""
import os
import glob
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("CMH_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "datasets"))
TSB_DIR = os.path.join(DATA_ROOT, "TSB-AD-MV", "TSB-AD-M")


def _zscore(X_train, X_all):
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_all - mu) / sigma


def load_tsb_ad(category, stream_id=1):
    """加载 TSB-AD-M 的某个子数据集.

    Args:
        category: 子数据集类别, 如 'Exathlon', 'Genesis', 'Daphnet', 'GHL', 'MSL'
        stream_id: stream 编号 (从1开始), 不同数据集有不同数量的 stream

    Returns:
        X: (T, D) float32, 已用训练块统计做 z-score
        Y: (T,) int8, 异常标签
    """
    # 找匹配的文件: NNN_Category_id_X_*.csv
    pattern = os.path.join(TSB_DIR, f"*_{category}_id_{stream_id}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        avail = glob.glob(os.path.join(TSB_DIR, f"*_{category}_*.csv"))
        n_streams = len(avail)
        raise FileNotFoundError(
            f"未找到 {category} id_{stream_id}. 该类别有 {n_streams} 个 stream. "
            f"可用 id 范围: 1-{n_streams}")

    fpath = matches[0]
    df = pd.read_csv(fpath)

    # 最后一列是 Label
    if 'Label' not in df.columns:
        Y_col = df.columns[-1]
    else:
        Y_col = 'Label'

    Y = df[Y_col].values.astype(np.int8)
    X = df.drop(columns=[Y_col]).select_dtypes(include=[np.number]).values.astype(np.float32)

    # 处理 NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 四块分割: train 35% 用于 z-score 统计
    T = len(X)
    n_tr = int(T * 0.35)
    X = _zscore(X[:n_tr], X)

    return X, Y


def list_tsb_ad_streams(category):
    """列出某类别的所有可用 stream 编号."""
    matches = glob.glob(os.path.join(TSB_DIR, f"*_{category}_*.csv"))
    ids = []
    for m in matches:
        try:
            # 文件名: NNN_Category_id_X_*.csv
            fname = os.path.basename(m)
            id_part = fname.split("_id_")[1].split("_")[0]
            ids.append(int(id_part))
        except (IndexError, ValueError):
            pass
    return sorted(set(ids))


if __name__ == "__main__":
    # 测试
    for cat in ["Exathlon", "Genesis", "Daphnet", "GHL", "MSL", "SMAP", "SMD"]:
        try:
            ids = list_tsb_ad_streams(cat)
            if ids:
                X, Y = load_tsb_ad(cat, ids[0])
                print(f"{cat:12s} streams={len(ids):3d}  id={ids[0]}: X={X.shape} D={X.shape[1]} ratio={Y.mean():.4f}")
        except Exception as e:
            print(f"{cat:12s} ERROR: {e}")
