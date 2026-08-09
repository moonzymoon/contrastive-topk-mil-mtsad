"""H1实验: TAIL-MIL 公平弱监督对比 (注入正bag)
标准MTSAD切分下weak-train全normal, TAIL-MIL退化.
本实验: 从test块借少量异常注入weak-train, 构造真正含正bag的弱监督协议,
让TAIL-MIL和CMH-MIL在公平条件下(都有正bag)对比."""
import sys, os, json, time, numpy as np, torch
import torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
PROJ = os.path.dirname(SRC)
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
from baselines_ext.tailmil_official import TAILMIL
from data.loaders import load_smd, load_nasa
from data.split import make_bags, bags_to_array
from sklearn.metrics import average_precision_score

SEEDS = [7, 42, 123, 202, 999]
L_BAG, L_INST, STRIDE = 128, 16, 64
ALPHA, EPOCHS, LR, BS = 0.05, 12, 0.002, 32


def build_bag_dataset(X, Y, start, end):
    """对 [start:end] 区间构造 bag, 返回 (Xb:(N,D,L), yb:(N,))."""
    seg_X = X[start:end]; seg_Y = Y[start:end]
    bags = make_bags(seg_X, seg_Y, L=L_BAG, l=L_INST, stride=STRIDE)
    if not bags: return None, None
    Xb, yb = bags_to_array(bags)
    N, D = Xb.shape[0], Xb.shape[-1]
    Xmodel = np.zeros((N, D, L_BAG), dtype=np.float32)
    for i in range(N):
        insts = Xb[i]; n = insts.shape[0]
        for k_idx in range(n):
            s = k_idx * STRIDE; e = s + L_INST
            if e <= L_BAG: Xmodel[i, :, s:e] = insts[k_idx].T
    return Xmodel, yb


def run_fair_tailmil(X, Y, D, seed, borrow_ratio=0.1):
    """公平弱监督: weak-train + 借用的正bag."""
    np.random.seed(seed); torch.manual_seed(seed)
    T = len(X)
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)
    # weak-train + 借用区(从test前段借异常)
    # test块: [c:T], 评估只用后半 [c+(T-c)//2 : T], 前半作借用源
    borrow_start = c; borrow_end = c + (T - c) // 2
    eval_start = borrow_end

    Xb_tr, yb_tr = build_bag_dataset(X, Y, 0, a)
    Xb_borrow, yb_borrow = build_bag_dataset(X, Y, borrow_start, borrow_end)
    Xb_cal, yb_cal = build_bag_dataset(X, Y, b, c)
    Xb_te, yb_te = build_bag_dataset(X, Y, eval_start, T)
    if any(x is None for x in [Xb_tr, Xb_borrow, Xb_cal, Xb_te]):
        return None

    # 借用区的正bag注入weak-train
    if yb_borrow.sum() > 0:
        pos_idx = np.where(yb_borrow == 1)[0]
        n_borrow = min(len(pos_idx), max(1, len(Xb_tr) // 10))
        rng = np.random.RandomState(seed)
        borrow = rng.choice(pos_idx, n_borrow, replace=False)
        Xb_tr = np.vstack([Xb_tr, Xb_borrow[borrow]])
        yb_tr = np.concatenate([yb_tr, yb_borrow[borrow]])

    model = TAILMIL(window=L_BAG, dim=D, sub_window=L_INST)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    Xt = torch.from_numpy(Xb_tr); yt = torch.from_numpy(yb_tr.astype(np.float32))
    N = Xt.shape[0]
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(N)
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            out, _, _ = model(Xt[idx]); out = out.squeeze(-1).clamp(1e-6, 1-1e-6)
            loss = loss_fn(out, yt[idx]); opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out_cal, _, _ = model(torch.from_numpy(Xb_cal)); sc = out_cal.squeeze(-1).numpy()
        out_te, contrib, _ = model(torch.from_numpy(Xb_te)); ste = out_te.squeeze(-1).numpy()
    normal_cal = sc[yb_cal == 0] if (yb_cal == 0).any() else sc
    n_cal = len(normal_cal); q = min(1.0, np.ceil((n_cal+1)*(1-ALPHA))/n_cal)
    tau = np.quantile(normal_cal, q) if n_cal > 0 else np.median(ste)
    if tau <= 0 or np.allclose(sc, 0):
        nz = normal_cal[normal_cal > 0]
        tau = np.quantile(nz, 0.95) if len(nz) > 0 else np.quantile(ste, 0.95)

    # bag分数->点分数
    Tte = len(X[eval_start:T]); ps = np.zeros(Tte, dtype=np.float32)
    bs = 0; bi = 0
    while bs + L_BAG <= Tte and bi < len(ste):
        ps[bs:bs+L_BAG] = ste[bi]; bs += STRIDE; bi += 1
    Yte = Y[eval_start:T]
    try:
        auprc = average_precision_score(Yte, ps)
    except:
        auprc = float('nan')
    pred = (ps > tau).astype(int)
    far = float(pred[Yte == 0].mean()) if (Yte == 0).any() else float('nan')
    return {'AUPRC': auprc, 'FAR': far, 'tau': float(tau), 'seed': seed,
            'n_pos_train': int(yb_tr.sum()), 'n_neg_train': int((yb_tr == 0).sum())}


def main():
    results = {}
    for name, loader in [('SMD', lambda: load_smd('machine-1-1')),
                         ('SMAP', lambda: load_nasa('SMAP'))]:
        print(f"\n=== {name} TAIL-MIL 公平弱监督 ===")
        X, Y = loader(); D = X.shape[1]
        per = []
        for seed in SEEDS:
            t0 = time.time()
            try:
                r = run_fair_tailmil(X, Y, D, seed)
                if r:
                    per.append(r)
                    print(f"  seed{seed}: AUPRC={r['AUPRC']:.4f} FAR={r['FAR']:.4f} 正bag={r['n_pos_train']} ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  seed{seed} fail: {e}")
        if per:
            agg = {k: f"{np.mean([r[k] for r in per if not np.isnan(r.get(k, float('nan')))]):.4f}±{np.std([r[k] for r in per if not np.isnan(r.get(k, float('nan')))]):.4f}"
                   for k in ['AUPRC', 'FAR'] if per}
            results[name] = {'per_seed': per, 'aggregate': agg}
            print(f"  AGG: {agg}")
    out = os.path.join(PROJ, 'results', 'tailmil_fair_weak.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\n保存 {out}")


if __name__ == '__main__':
    main()
