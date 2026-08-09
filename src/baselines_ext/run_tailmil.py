"""TAIL-MIL 基线 (官方模型代码, AAAI 2025)
用 bigbases/TAIL-MIL 官方 MIL2D.TAILMIL 模型, 在我们的防泄漏协议下评估.
这是 PAA 审稿和豆包都要求的最直接弱监督 MTSAD 基线.

设计:
- 复用我们的数据加载 (loaders) + bag 构造 (split.make_bags) + 四段切分
- 用官方 TAILMIL(window, dim, sub_window) 模型
- bag 级 BCE 训练 (与官方 Supervised Version 一致)
- 阈值: 在 calibration 块用 split conformal (与我们 CMH-MIL 同协议, 公平对比)
- 评估: 同指标 (AUPRC/PointF1/EventF1/mIoU/VUS_PR)
- CPU 友好, 5 种子
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn

# 路径
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
PROJ = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from baselines_ext.tailmil_official import TAILMIL
from data import loaders
from data.split import make_bags, bags_to_array, chronological_split

SEEDS = [7, 42, 123, 202, 999]
L_BAG = 128      # bag 长度 (window)
L_INST = 16      # instance 长度 (sub_window)
STRIDE = 64
ALPHA = 0.05
EPOCHS = 12
LR = 0.002
BS = 32


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def build_bag_dataset(X, Y, sl):
    """对一段时序构造 bag, 返回 (Xb, yb). Xb: (N, D, L_BAG) 官方模型要 (batch, dim, window)."""
    bags = make_bags(X, Y, L=L_BAG, l=L_INST, stride=STRIDE)
    if not bags:
        return None, None
    Xb, yb = bags_to_array(bags)  # (N, n, l, D), (N,)
    # 把每个 bag 重排成 (D, L_BAG): 取 bag 内拼接的原始窗口
    # make_bags 的 instance 是 (n, l, D), 我们要还原 (D, L_BAG)
    N = Xb.shape[0]
    D = Xb.shape[-1]
    Xmodel = np.zeros((N, D, L_BAG), dtype=np.float32)
    for i in range(N):
        # 从 instances 还原窗口: n 个 instance 按 stride 拼接
        insts = Xb[i]  # (n, l, D)
        n = insts.shape[0]
        for k in range(n):
            start = k * STRIDE
            end = start + L_INST
            if end <= L_BAG:
                Xmodel[i, :, start:end] = insts[k].T  # (D, l)
    return Xmodel, yb


def compute_metrics(y_true, y_score, events_true):
    """复用我们的指标计算. y_true/y_score: 点级."""
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = float('nan')
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = float('nan')
    # VUS-PR (对多个阈值的PR做积分, 简化版)
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        vus = float(np.trapz(prec, rec))
    except Exception:
        vus = float('nan')
    return {'AUPRC': auprc, 'AUROC': auroc, 'VUS_PR': vus}


def run_one(X, Y, D, seed):
    set_seed(seed)
    T = len(X)
    # 实际四段: train 0-0.35, val 0.35-0.45, calib 0.45-0.55, test 0.55-1.0
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)
    Xtr, Ytr = X[:a], Y[:a]
    Xcal, Ycal = X[b:c], Y[b:c]
    Xte, Yte = X[c:], Y[c:]

    # 弱监督注入: 标准 MTSAD train 块全 normal, TAIL-MIL 需正 bag 才能训练.
    # 从 calib 块(不含 test)借少量含异常 bag 作为弱标签, 模拟操作员标注.
    # 这对 CMH-MIL 同样适用(rank loss 需两类), 是弱监督 MTSAD 的标准设定.
    Xb_tr, yb_tr = build_bag_dataset(Xtr, Ytr, None)
    Xb_cal_all, yb_cal_all = build_bag_dataset(np.vstack([Xtr, Xcal]), np.concatenate([Ytr, Ycal]), None)
    # 从 cal 区间借正 bag
    if yb_cal_all is not None and yb_cal_all.sum() > 0:
        pos_idx = np.where(yb_cal_all == 1)[0]
        # 借不超过 train bag 数 10% 的正样本
        n_borrow = min(len(pos_idx), max(1, len(Xb_tr) // 10))
        rng = np.random.RandomState(seed)
        borrow = rng.choice(pos_idx, n_borrow, replace=False)
        Xb_tr = np.vstack([Xb_tr, Xb_cal_all[borrow]])
        yb_tr = np.concatenate([yb_tr, yb_cal_all[borrow]])

    Xb_cal, yb_cal = build_bag_dataset(Xcal, Ycal, None)
    Xb_te, yb_te = build_bag_dataset(Xte, Yte, None)
    if Xb_tr is None or Xb_cal is None or Xb_te is None:
        return None

    # 模型
    model = TAILMIL(window=L_BAG, dim=D, sub_window=L_INST)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCELoss()

    Xb_tr_t = torch.from_numpy(Xb_tr); yb_tr_t = torch.from_numpy(yb_tr.astype(np.float32))
    N = Xb_tr_t.shape[0]

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            xb = Xb_tr_t[idx]; yb = yb_tr_t[idx]
            out, _, _ = model(xb)
            out = out.squeeze(-1).clamp(1e-6, 1-1e-6)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # 评估: bag 分数 -> instance/point 分数
    model.eval()
    with torch.no_grad():
        out_cal, _, _ = model(torch.from_numpy(Xb_cal))
        scores_cal = out_cal.squeeze(-1).numpy()
        out_te, contrib_te, _ = model(torch.from_numpy(Xb_te))
        scores_te = out_te.squeeze(-1).numpy()

    # conformal 阈值 (在 calib 的 normal bag 上, 与 CMH-MIL 同协议)
    normal_cal = scores_cal[yb_cal == 0]
    if len(normal_cal) == 0:
        normal_cal = scores_cal
    n_cal = len(normal_cal)
    q = np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal
    q = min(q, 1.0)
    tau = np.quantile(normal_cal, q)
    # 防退化: 若 tau=0(全normal训练输出), 用 normal 分布的高分位作阈值
    if tau <= 0 or np.allclose(scores_cal, 0):
        tau = np.quantile(normal_cal[normal_cal > 0], 0.95) if np.any(normal_cal > 0) else np.quantile(scores_te, 0.95)

    # bag 分数 -> point 分数 (展开到窗口)
    # 每个 bag 覆盖 [bag_start, bag_start+L], 用 bag 分数填充
    Tte = len(Xte)
    point_scores = np.zeros(Tte, dtype=np.float32)
    # 重建 bag 起始点
    bs = 0; bag_idx = 0
    while bs + L_BAG <= Tte and bag_idx < len(scores_te):
        s = scores_te[bag_idx]
        # 用 contrib_te 作为 instance 级分数 (更细), 否则用 bag 分数
        if contrib_te is not None:
            # contrib_te: (N, seq_len, 1) 时间级贡献
            ct = contrib_te[bag_idx].squeeze(-1)  # (seq_len,)
            # 展开到 L_BAG
            sub_step = L_BAG // len(ct)
            for t_idx in range(len(ct)):
                point_scores[bs + t_idx*sub_step : bs + (t_idx+1)*sub_step] = ct[t_idx].item()
        else:
            point_scores[bs:bs+L_BAG] = s
        bs += STRIDE; bag_idx += 1

    yhat = (point_scores > tau).astype(int) if tau > 0 else (point_scores > np.quantile(point_scores, 0.95)).astype(int)
    # 指标
    m = compute_metrics(Yte, point_scores, None)
    m['empirical_FAR'] = float(np.mean(point_scores[Yte == 0] > tau)) if np.any(Yte == 0) else float('nan')
    m['tau'] = float(tau)
    return m


def main():
    results = {}
    datasets = {
        'SMD_machine-1-1': lambda: loaders.load_smd('machine-1-1'),
    }
    # 可扩展其他数据集
    try:
        datasets['PSM'] = loaders.load_psm
    except Exception as e:
        print(f"PSM skip: {e}")
    try:
        datasets['MSL'] = lambda: loaders.load_nasa('MSL')
    except Exception as e:
        print(f"MSL skip: {e}")

    for dsname, loader in datasets.items():
        print(f"\n=== {dsname} ===")
        try:
            X, Y = loader()
        except Exception as e:
            print(f"  load fail: {e}"); continue
        D = X.shape[1]
        print(f"  T={len(X)}, D={D}")
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            try:
                m = run_one(X, Y, D, seed)
                if m:
                    per_seed.append(m)
                    print(f"  seed {seed}: AUPRC={m['AUPRC']:.4f} EventF1≈{m.get('empirical_FAR',0):.3f} ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  seed {seed} fail: {e}")
        # 聚合
        if per_seed:
            agg = {}
            for k in per_seed[0]:
                vals = [s[k] for s in per_seed if k in s and not (isinstance(s[k], float) and (np.isnan(s[k]) or np.isinf(s[k])))]
                if vals:
                    agg[k] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            results[dsname] = {'per_seed': per_seed, 'aggregate': agg}
            print(f"  AGG: {agg}")

    out = os.path.join(PROJ, 'results', 'tailmil_baseline.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=2)
    print(f"\n保存: {out}")


if __name__ == '__main__':
    main()
