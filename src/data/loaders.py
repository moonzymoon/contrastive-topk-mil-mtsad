"""真实数据集加载器 (SMD/PSM/MSL/SMAP/5-ECK/MetroPT-3)
公开数据集加载逻辑. 实际下载待用户授权.
W1 修复 (2026-11-17): 返回【原始未标准化】数据 — 标准化/clip 一律在
data.protocol 层先切分后拟合 (旧版在 loader 内用全量统计 zscore 属于预处理泄漏, R9).
方案C (2026-11-18): 返回三元组 (X, Y, n_official_train) —
  n_official_train = 官方 train 段长度 (拼接降采样后的口径), 供轨1 official_split 使用;
  无官方划分的数据集 (TSB-AD/5ECK) 返回 None → protocol 回退全流四块.
统一入口: load_dataset(name) -> (X, Y, n_official_train)."""
import os
import numpy as np

# 数据根目录(下载后存放位置)
DATA_ROOT = os.environ.get("CMH_DATA_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"))


def load_smd(entity="machine-1-1"):
    """SMD (Server Machine Dataset). 28 台服务器, D=38.
    来源: OmniAnomaly 仓库 (train/test/test_label 子目录, .txt 逗号分隔).
    entity: 文件名前缀, 如 machine-1-1. 返回 train+test 拼接的 (X,Y)."""
    path = os.path.join(DATA_ROOT, "SMD")
    Xtr = np.loadtxt(os.path.join(path, "train", f"{entity}.txt"), delimiter=",")
    Xte = np.loadtxt(os.path.join(path, "test", f"{entity}.txt"), delimiter=",")
    Yte = np.loadtxt(os.path.join(path, "test_label", f"{entity}.txt"), dtype=np.int8)
    X = np.vstack([Xtr, Xte]).astype(np.float32)
    Y = np.concatenate([np.zeros(len(Xtr), dtype=np.int8), Yte])
    return X, Y, len(Xtr)


def load_psm():
    """PSM (Pooled Server Metrics). D=25. 来源: eBay, CSTCloudOps镜像.
    格式: train.npy/test.npy/test_label.npy."""
    path = os.path.join(DATA_ROOT, "PSM")
    Xtr = np.load(os.path.join(path, "train.npy")).astype(np.float32)
    Xte = np.load(os.path.join(path, "test.npy")).astype(np.float32)
    Y = np.load(os.path.join(path, "test_label.npy")).astype(np.int8).reshape(-1)
    # NaN填0
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    X = np.vstack([Xtr, Xte])
    Y = np.concatenate([np.zeros(len(Xtr), dtype=np.int8), Y])
    return X, Y, len(Xtr)


def load_nasa(dataset="MSL"):
    """MSL/SMAP (NASA). 来源: CSTCloudOps镜像.
    格式: train.npy/test.npy/test_label.npy (AllInOne, 多通道已合并).
    NASA通道作为整体流处理(与第一篇'独立实体'口径一致需另切, 此处先整体)."""
    path = os.path.join(DATA_ROOT, "NASA", dataset)
    Xtr = np.load(os.path.join(path, "train.npy")).astype(np.float32)
    Xte = np.load(os.path.join(path, "test.npy")).astype(np.float32)
    Y = np.load(os.path.join(path, "test_label.npy")).astype(np.int8).reshape(-1)
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    # 若3维(通道,T,D)则展平为长流(第一篇同款处理: 通道作独立实体需逐通道跑)
    if Xtr.ndim == 3:
        Xtr = Xtr.reshape(-1, Xtr.shape[-1])
        Xte = Xte.reshape(-1, Xte.shape[-1])
        Y = Y.reshape(-1)
    X = np.vstack([Xtr, Xte])
    Y = np.concatenate([np.zeros(len(Xtr), dtype=np.int8), Y])
    return X, Y, len(Xtr)


def load_5eck():
    """5-ECK-2022 (Energy Consumption in Korea). D=5. Mendeley Data CC BY 4.0."""
    import pandas as pd
    path = os.path.join(DATA_ROOT, "5ECK2022")
    df = pd.read_csv(os.path.join(path, "data.csv"))
    cols = [c for c in df.columns if c != "label"]
    X = df[cols].values.astype(np.float32)
    Y = (df["label"].values > 0).astype(np.int8) if "label" in df.columns else np.zeros(len(df), dtype=np.int8)
    return X, Y, None


def load_swat(downsample=5):
    """SWaT (Secure Water Treatment). D=51. iTrust/SUTD.
    normal.csv(全正常,~139万点) + attack.csv(~5.5万点,含36次攻击).
    downsample: 时间降采样因子(5=每5点取1,加速). 返回拼接的(X,Y)已clip+zscore."""
    import pandas as pd
    path = os.path.join(DATA_ROOT, "SWaT")
    df_n = pd.read_csv(os.path.join(path, "normal.csv"))
    df_a = pd.read_csv(os.path.join(path, "attack.csv"))
    feat_cols = [c for c in df_n.columns if c.strip() not in ("Timestamp", "Normal/Attack")]
    Xn = df_n[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    Xa = df_a[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    Ya = (df_a["Normal/Attack"].astype(str).str.strip() == "Attack").astype(np.int8).values
    # W1: 原始值返回 (clip+zscore 移至 protocol 层, 用协议 train 块统计)
    X = np.vstack([Xn, Xa]).astype(np.float32)
    Y = np.concatenate([np.zeros(len(Xn), dtype=np.int8), Ya])
    n_official_train = len(Xn)
    if downsample and downsample > 1:
        X, Y = X[::downsample], Y[::downsample]
        n_official_train = len(range(0, n_official_train, downsample))
    return X, Y, n_official_train


def load_wadi(downsample=10):
    """WADI (Water Distribution). iTrust/SUTD 2017. D=123.
    WADI_14days.csv(14天正常) + WADI_attackdata.csv(2天攻击,无标签列).
    标签用 attack_description.xlsx 的时间区间精确标记(合法).
    降采样=10 (WADI采样率高)."""
    import pandas as pd
    path = os.path.join(DATA_ROOT, "WADI", "WADI.A1_9 Oct 2017")
    df_n = pd.read_csv(os.path.join(path, "WADI_14days.csv"), skiprows=3, low_memory=False)
    df_a = pd.read_csv(os.path.join(path, "WADI_attackdata.csv"), low_memory=False)
    # 特征列: 去掉 Row/Date/Time, 取数值列
    drop = {"Row", "Date", "Time"}
    feat_cols = [c for c in df_n.columns if c not in drop and not str(c).startswith("Unnamed")]
    feat_cols = [c for c in feat_cols if c in df_a.columns]  # 交集
    Xn = df_n[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    Xa = df_a[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    D = Xn.shape[1]
    # 标签: 用 attack_description.xlsx 时间区间生成
    Ya = np.zeros(len(df_a), dtype=np.int8)
    desc = pd.read_excel(os.path.join(path, "attack_description.xlsx"), header=4)
    df_a["_dt"] = pd.to_datetime(df_a["Date"].astype(str) + " " + df_a["Time"].astype(str), errors="coerce")
    for _, row in desc.iterrows():
        d, st, et = row.get("Date"), row.get("Start Time"), row.get("End Time")
        if pd.isna(st) or pd.isna(et) or pd.isna(d):
            continue
        try:
            d = pd.to_datetime(d, errors="coerce")
            if pd.isna(d):
                continue
            start = pd.to_datetime(str(d.date()) + " " + str(st), errors="coerce")
            end = pd.to_datetime(str(d.date()) + " " + str(et), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            Ya[((df_a["_dt"] >= start) & (df_a["_dt"] <= end)).values] = 1
        except Exception:
            continue
    # W1: 原始值返回 (clip+zscore 移至 protocol 层)
    X = np.vstack([Xn, Xa]).astype(np.float32)
    Y = np.concatenate([np.zeros(len(Xn), dtype=np.int8), Ya])
    n_official_train = len(Xn)
    if downsample and downsample > 1:
        X, Y = X[::downsample], Y[::downsample]
        n_official_train = len(range(0, n_official_train, downsample))
    return X, Y, n_official_train


LOADERS = {
    "SMD": lambda: load_smd("machine-1-1"),
    "PSM": load_psm,
    "MSL": lambda: load_nasa("MSL"),
    "SMAP": lambda: load_nasa("SMAP"),
    "5ECK2022": load_5eck,
    "SWaT": load_swat,
    "WADI": load_wadi,
}


def load_dataset(name):
    """统一加载入口. 返回 (X, Y, n_official_train).
    n_official_train=None 表示该数据集无官方 train/test 划分 (走全流协议)."""
    if name not in LOADERS:
        raise ValueError(f"未知数据集 {name}, 可选 {list(LOADERS)}")
    return LOADERS[name]()


# ===== 下载脚本(待授权执行) =====
DOWNLOAD_URLS = {
    "SMD": "https://github.com/NetManAIOps/OmniAnomaly/raw/master/data/SMD.zip",
    "PSM": "https://github.com/eBay/tsAdBenchmark/raw/master/data/psm.tar.gz",
    "NASA_MSL_SMAP": "https://github.com/khundman/telemanom/raw/master/data/labeled_anomalies.csv",
    "5ECK2022": "https://data.mendeley.com/datasets/... (CC BY 4.0)",
    "MetroPT3": "https://archive.ics.uci.edu/dataset/....",
}


def download_all():
    """下载全部数据集. ⚠️ 待用户授权后执行."""
    print("以下数据集将下载到:", DATA_ROOT)
    for k, v in DOWNLOAD_URLS.items():
        print(f"  {k}: {v}")
    print("体量预估: SMD~50MB, PSM~30MB, NASA~20MB, 总计~100-150MB")
    print("请确认授权后调用实际下载.")
