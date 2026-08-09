"""TopK 方差证据实验 (直接复用 run_v2 训练逻辑, 加 w_topk 开关).
对比有/无 TopK 时, AUPRC 和 Event-F1 在5种子上的方差.
审稿人质疑 TopK 对 AUPRC 无影响, 这里证明它稳定 Event-F1 方差.
"""
import sys, os, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from utils.repro import load_config, set_seed
from data.loaders import load_dataset
from data.split import make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss_v2 import CMHLossV2
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all

SEEDS = [7, 42, 123, 202, 999]


def run_v2_with_topk_flag(X, Y, cfg, seed, w_topk):
    """复用 run_v2 训练逻辑, w_topk 可调 (0=关闭TopK, 1=开启)."""
    set_seed(seed)
    D = X.shape[1]; T = len(X)
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)
    mk = lambda s, e: make_bags(X[s:e], Y[s:e], L=cfg['bag']['L'], l=cfg['bag']['l'], stride=cfg['bag']['stride'])
    wX, wY = bags_to_array(mk(0, a))
    cX, cY = bags_to_array(mk(b, c))
    tX, tY = bags_to_array(mk(c, T))

    k = cfg['model']['hyperspheres']['k']
    model = CMHMIL(D, d_embed=cfg['model']['encoder']['d_embed'], k=k,
                   channels=tuple(cfg['model']['encoder']['channels']),
                   kernel_size=cfg['model']['encoder']['kernel_size'],
                   dropout=cfg['model']['encoder']['dropout'],
                   use_var_attn=cfg['model']['var_attention'], pool=cfg['conformal']['bag_pool'])
    crit = CMHLossV2(w_bag=1.0, w_compact=0.10, w_sep=0.05, w_topk=w_topk,
                     w_contrast=0.5, w_balance=0.10, topk_margin=1.0, contrast_tau=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    wXt = torch.from_numpy(wX); wYt = torch.from_numpy(wY.astype(np.float32))
    n_init = min(256, len(wXt))
    model.init_centers_kmeans(wXt[torch.randperm(len(wXt))[:n_init]], 'cpu')
    bs = cfg['train']['batch_size']

    for ep in range(cfg['train']['max_epochs']):
        model.train(); perm = torch.randperm(len(wXt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            if len(idx) < 4: continue
            xb, yb = wXt[idx], wYt[idx]
            scores, _, z = model(xb, return_z=True)
            z_flat = z.reshape(len(idx), -1, z.shape[-1])
            bag_z = z_flat.mean(dim=1)
            loss, parts = crit(scores, yb, z.reshape(-1, z.shape[-1]), bag_z,
                               model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip']); opt.step()

    model.eval()
    with torch.no_grad():
        cs, _ = model(torch.from_numpy(cX)); ts, _ = model(torch.from_numpy(tX))
    def pool(arr):
        w = np.exp(arr - arr.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return (w * arr).sum(1)
    cBag, tBag = pool(cs.cpu().numpy()), pool(ts.cpu().numpy())
    cBag = np.sign(cBag) * np.log1p(np.abs(cBag) * 0.1)
    tBag = np.sign(tBag) * np.log1p(np.abs(tBag) * 0.1)
    thr = split_conformal_threshold(cBag[cY == 0], alpha=cfg['conformal']['alpha'])
    t_pred = (tBag > thr).astype(int)
    m = evaluate_all(tY, tBag, thr, events_from_binary(tY), events_from_binary(t_pred))
    m['empirical_FAR'] = float(t_pred[(tY == 0)].mean()) if (tY == 0).sum() > 0 else float('nan')
    return m


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))
    print('=== TopK 方差证据实验 (MSL, 5 seeds) ===\n')
    X, Y = load_dataset('MSL')
    print(f'MSL: X={X.shape}, ratio={Y.mean():.4f}\n')

    results = {}
    for w_topk, label in [(1.0, 'with_topk'), (0.0, 'without_topk')]:
        print(f'[{"WITH" if w_topk > 0 else "WITHOUT"} TopK (w_topk={w_topk})]')
        per_seed = []
        for seed in SEEDS:
            try:
                m = run_v2_with_topk_flag(X, Y, cfg, seed, w_topk)
                m['seed'] = seed
                per_seed.append(m)
                print(f'  seed {seed}: AUPRC={m["AUPRC"]:.4f} EF1={m["EventF1_one2one"]:.4f}')
            except Exception as e:
                print(f'  seed {seed} FAILED: {e}')
        results[label] = per_seed
        print()

    # 分析
    print('='*60)
    print('方差分析 (TopK 稳定 Event-F1 的证据)')
    print('='*60)
    for metric in ['AUPRC', 'EventF1_one2one', 'empirical_FAR']:
        w_vals = [s[metric] for s in results['with_topk'] if metric in s]
        wo_vals = [s[metric] for s in results['without_topk'] if metric in s]
        if w_vals and wo_vals:
            wm, ws = statistics.mean(w_vals), statistics.stdev(w_vals)
            wom, wos = statistics.mean(wo_vals), statistics.stdev(wo_vals)
            print(f'{metric}:')
            print(f'  WITH TopK:    {wm:.4f} ± {ws:.4f}')
            print(f'  WITHOUT TopK: {wom:.4f} ± {wos:.4f}')
            if wos > 0 and metric == 'EventF1_one2one':
                reduction = (1 - ws/wos) * 100
                print(f'  → Event-F1 标准差降低 {reduction:.1f}%')
            print()

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'topk_variance.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'保存到 {out}')


if __name__ == '__main__':
    main()
