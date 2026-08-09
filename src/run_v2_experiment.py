"""CMH-MIL v2 实验：对比学习 + TopK + 注意力归因
在 SMD/PSM/MSL/SMAP 上跑 v2 vs v1 vs 基线, 5种子."""
import sys, os, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
from utils.repro import load_config, set_seed
from data.loaders import load_dataset
from data.split import chronological_split, make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss_v2 import CMHLossV2
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all

SEEDS = [7, 42, 123, 202, 999]


def run_v2(X, Y, cfg, seed, dataset_name):
    """跑 v2 (对比学习+TopK)."""
    set_seed(seed)
    D = X.shape[1]
    T = len(X)
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
    # v2 损失
    crit = CMHLossV2(w_bag=1.0, w_compact=0.10, w_sep=0.05, w_topk=1.0,
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
            if len(idx) < 4: continue  # 对比学习需足够 batch
            xb, yb = wXt[idx], wYt[idx]
            scores, _, z = model(xb, return_z=True)
            # bag 级表征 (池化 z)
            z_flat = z.reshape(len(idx), -1, z.shape[-1])  # (B, n, d)
            bag_z = z_flat.mean(dim=1)  # (B, d) mean pool 作对比学习输入
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
    # v2关键修复: 有界分数变换(log1p), 防止test分数爆炸导致共形阈值失效
    cBag = np.sign(cBag) * np.log1p(np.abs(cBag) * 0.1)
    tBag = np.sign(tBag) * np.log1p(np.abs(tBag) * 0.1)
    thr = split_conformal_threshold(cBag[cY == 0], alpha=cfg['conformal']['alpha'])
    t_pred = (tBag > thr).astype(int)
    m = evaluate_all(tY, tBag, thr, events_from_binary(tY), events_from_binary(t_pred))
    m['empirical_FAR'] = float(t_pred[(tY == 0)].mean()) if (tY == 0).sum() > 0 else float('nan')
    return m


def main():
    cfg = load_config(os.path.join(PROJ, 'configs', 'default.yaml'))
    DATASETS = ["SMD", "PSM", "MSL", "SMAP"]
    results = {}
    for ds in DATASETS:
        try:
            X, Y = load_dataset(ds)
            print(f"\n=== {ds}: X{X.shape} anomaly={Y.mean():.4f} ===")
        except Exception as e:
            print(f"{ds}: skip ({e})"); continue
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            try:
                m = run_v2(X, Y, cfg, seed, ds)
                m['seed'] = seed; per_seed.append(m)
                print(f"  seed{seed}: AUPRC={m['AUPRC']:.4f} EventF1={m['EventF1_one2one']:.4f} FAR={m['empirical_FAR']:.4f} ({time.time()-t0:.0f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  seed{seed} fail: {e}")
        agg = {}
        for kk in ['AUPRC', 'AUROC', 'PointF1', 'EventF1_one2one', 'AllEventMIoU', 'VUS_PR', 'empirical_FAR']:
            vals = [r[kk] for r in per_seed if kk in r and not (isinstance(r[kk], float) and (np.isnan(r[kk]) or np.isinf(r[kk])))]
            if vals:
                agg[kk] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
        results[ds] = {'per_seed': per_seed, 'aggregate': agg}
        print(f"  AGG: {agg}")
    out = os.path.join(PROJ, 'results', 'v2_experiment.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=2)
    print(f"\n保存 {out}")


if __name__ == '__main__':
    main()
