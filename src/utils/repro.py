"""可复现性工具: 种子 / 配置 / 哈希 / 日志
合规强制: 固定种子, 配置归档, 文件哈希, 完整日志, 阴性结果留存."""
import os, json, hashlib, random, logging, sys
from datetime import datetime
from pathlib import Path


def set_seed(seed: int):
    """固定全部随机源."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_sha256(path) -> str:
    """计算文件 SHA-256 哈希, 用于完整性溯源."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(log_dir: str, seed: int, dataset: str) -> logging.Logger:
    """完整训练日志, 含起止时间/失败原因."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{dataset}_seed{seed}_{ts}.log")
    logger = logging.getLogger(f"{dataset}_{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"=== CMH-MIL run | dataset={dataset} seed={seed} | start={ts} ===")
    logger.info(f"log_file={log_path}")
    return logger


def save_run_meta(out_dir: str, config: dict, seed: int, dataset: str, status: str, note: str = ""):
    """保存运行元数据(配置/状态/备注). status 含 success/fail/aborted."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "dataset": dataset, "seed": seed, "status": status,
        "timestamp": ts, "note": note,
        "config": config,
    }
    p = os.path.join(out_dir, f"meta_{dataset}_seed{seed}_{ts}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p


def load_config(path: str) -> dict:
    """加载 YAML 配置."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
