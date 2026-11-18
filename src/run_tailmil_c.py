"""TAIL-MIL 轨2适配 (方案C任务书步骤2, 2026-11-18)
==========================================================
同预算弱监督对比: TAIL-MIL (官方模型, AAAI 2025) 消费与 v3 相同的 annot 块
bag 标签 (同预算), 走 plan-c-annot 协议 + 统一点级管线 (instance→max→阈值→点级指标).

与旧 run_tailmil.py 的区别 (三外审 A1/R2-3):
  - 旧: four-block 手写切分 + 从 calib/test 借异常袋 (标签预算不公平, 借test=泄漏)
  - 新: official_split + annot 块并入训练 (与 v3-annot 完全同标签预算)
  - 旧: bag 分数粗暴展开; 新: contrib 时间级分数 → point_scores(max) 统一管线
  - 阈值: 统一 threshold_from_calib (calib 正常点 Q95), 不再手写退化兜底
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "baselines_ext"))
import numpy as np
import torch
import torch.nn as nn

from data.protocol import (weak_annotation_split, Scaler, make_bags_v2,
                           point_scores, threshold_from_calib, evaluate_point)
from baselines_ext.tailmil_official import TAILMIL

L, l, BAG_STRIDE = 128, 16, 64
EPOCHS, LR, BS, ALPHA = 12, 2e-3, 32, 0.05


def _to_model_input(insts):
    """make_bags_v2 的 (N, n, l, D) (n=8 无缝覆盖) → 官方模型输入 (N, D, L).
    n*l == L, instance 无重叠 → instance 轴拼接即原窗口 (无损重排)."""
    N, n, ll, D = insts.shape
    assert n * ll == L, f"n*l={n*ll} != L={L}"
    return insts.transpose(0, 3, 1, 2).reshape(N, D, L)


def tailmil_plan_c(X, Y, n_official_train, seed=7, epochs=EPOCHS):
    """TAIL-MIL 在 plan-c-annot 协议下单种子评估. 返回与 run_v3 同构指标 dict."""
    np.random.seed(seed); torch.manual_seed(seed)
    sp = weak_annotation_split(len(Y), n_official_train)
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)
    D = X.shape[1]

    blocks = {}
    for name in ("train", "calib", "test", "annot"):
        s, e = sp[name]
        insts, y_bag, starts, rel = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
        blocks[name] = (insts, y_bag, starts + s, rel)

    # 训练集 = 轨1 train + annot 块 (同预算; annot bag 标签与 v3-annot 相同)
    tr_insts = np.concatenate([blocks["train"][0], blocks["annot"][0]])
    tr_y = np.concatenate([blocks["train"][1], blocks["annot"][1]])
    Xm_tr = _to_model_input(tr_insts)

    model = TAILMIL(window=L, dim=D, sub_window=l)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    Xt = torch.from_numpy(Xm_tr.astype(np.float32))
    Yt = torch.from_numpy(tr_y.astype(np.float32))

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            out, _, _ = model(Xt[idx])
            out = out.squeeze(-1).clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(out, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # 推理: contrib (N, seq_len=8, 1) 作为 instance 级分数 → 统一点级管线
    s_all = np.full(len(Y), np.nan, dtype=np.float32)
    with torch.no_grad():
        for name in ("calib", "test"):
            insts, yb, starts, rel = blocks[name]
            Xm = _to_model_input(insts)
            N, n = insts.shape[0], insts.shape[1]
            sc = []
            for i in range(0, N, 64):
                _, contrib, _ = model(torch.from_numpy(Xm[i:i + 64].astype(np.float32)))
                sc.append(contrib.squeeze(-1).numpy())          # (B, 8)
            sc = np.concatenate(sc)
            s_blk, _ = point_scores(len(Y), starts, rel, sc, l, agg="max")
            s_all[sp[name][0]:sp[name][1]] = s_blk[sp[name][0]:sp[name][1]]

    thr = threshold_from_calib(s_all[sp["calib"][0]:sp["calib"][1]],
                               Y[sp["calib"][0]:sp["calib"][1]], alpha=ALPHA)
    cov = ~np.isnan(s_all[sp["test"][0]:sp["test"][1]])
    m = evaluate_point(Y[sp["test"][0]:sp["test"][1]],
                       s_all[sp["test"][0]:sp["test"][1]], thr, cov)
    m["threshold"] = thr
    m["train_bags"] = int(len(tr_y))
    m["train_anom_bags"] = int(tr_y.sum())
    m["annot_bags"] = int(len(blocks["annot"][1]))
    m["split_mode"] = "plan-c-annot"
    return m


if __name__ == "__main__":
    from run_v3 import load
    ds = sys.argv[1] if len(sys.argv) > 1 else "SMD"
    X, Y, n_off = load(ds)
    import time
    t0 = time.time()
    m = tailmil_plan_c(X, Y, n_off, seed=7)
    print(f"TAIL-MIL {ds} (轨2): AUPRC={m['AUPRC']:.4f} EF1={m['EventF1_one2one']:.4f} "
          f"FAR={m['FAR_point']:.4f} ({time.time()-t0:.0f}s)")
