"""补充实验脚本:
1. Ours + PCA + AT 在新数据集 (Genesis/Daphnet/GHL) 上跑
2. AT baseline 补到 SWaT/WADI/Exathlon-1/2
3. TopK 方差证据: 有/无 TopK 的 5 种子 Event-F1 对比 (MSL)

复用项目的四块分割协议和评估指标.
"""
import sys, os, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data.loaders import load_dataset
from data.tsb_ad_loader import load_tsb_ad
from baselines import baseline_pca, baseline_autoencoder, baseline_isolation_forest
from baseline_anomaly_transformer import baseline_anomaly_transformer
from run_v2_experiment import run_v2
from utils.repro import load_config

SEEDS = [7, 42, 123, 202, 999]
CFG = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))


def aggregate(per_seed):
    """聚合 per_seed 结果为 mean±std."""
    agg = {}
    for key in ['AUPRC', 'AUROC', 'PointF1', 'EventF1_one2one', 'empirical_FAR']:
        vals = [s[key] for s in per_seed if key in s and s[key] == s[key]]
        if vals:
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            agg[key] = f'{m:.4f}±{sd:.4f}'
    return agg


def run_method(method_fn, X, Y, seeds, cfg=None, **kwargs):
    """对某方法跑多种子, 返回 per_seed list. 自动检测 seed 参数."""
    import inspect
    per_seed = []
    accepts_seed = 'seed' in inspect.signature(method_fn).parameters
    for seed in seeds:
        try:
            if accepts_seed:
                m = method_fn(X, Y, cfg or {}, seed=seed, **kwargs)
            else:
                np.random.seed(seed)
                m = method_fn(X, Y, cfg or {}, **kwargs)
            m['seed'] = seed
            per_seed.append(m)
            print(f'    seed {seed}: AUPRC={m["AUPRC"]:.4f} EF1={m.get("EventF1_one2one",0):.4f}')
        except Exception as e:
            print(f'    seed {seed} FAILED: {e}')
    return per_seed


def main():
    results = {}

    # ============ Part 1: 新数据集 (Genesis/Daphnet/GHL) ============
    print('\n' + '='*60)
    print('PART 1: 新数据集 Ours + PCA + AT')
    print('='*60)

    new_datasets = {
        'Genesis':  ('tsb', 'Genesis', 1),
        'Daphnet':  ('tsb', 'Daphnet', 1),
        'CATSv2':   ('tsb', 'CATSv2', 1),
    }

    for name, (src, cat, sid) in new_datasets.items():
        print(f'\n--- {name} ({cat}) ---')
        try:
            if src == 'tsb':
                X, Y, _n = load_tsb_ad(cat, sid)
            else:
                X, Y, _n = load_dataset(cat)
            print(f'  X={X.shape}, Y={Y.shape}, anomaly_ratio={Y.mean():.4f}')
        except Exception as e:
            print(f'  加载失败: {e}')
            continue

        ds_res = {}
        # Ours
        print('  [Ours]')
        ps = run_method(run_v2, X, Y, SEEDS, cfg=CFG, dataset_name=name)
        if ps: ds_res['Ours'] = {'per_seed': ps, 'aggregate': aggregate(ps)}
        # PCA
        print('  [PCA]')
        ps = run_method(baseline_pca, X, Y, SEEDS, cfg=CFG)
        if ps: ds_res['PCA'] = {'per_seed': ps, 'aggregate': aggregate(ps)}
        # AT
        print('  [AT]')
        ps = run_method(baseline_anomaly_transformer, X, Y, SEEDS)
        if ps: ds_res['AT'] = {'per_seed': ps, 'aggregate': aggregate(ps)}
        # AE
        print('  [AE]')
        ps = run_method(baseline_autoencoder, X, Y, SEEDS, cfg=CFG)
        if ps: ds_res['AE'] = {'per_seed': ps, 'aggregate': aggregate(ps)}

        results[name] = ds_res

    # ============ Part 2: AT 补到 SWaT/WADI/Exathlon ============
    print('\n' + '='*60)
    print('PART 2: AT baseline 补全到 SWaT/WADI/Exathlon')
    print('='*60)

    at_datasets = {
        'SWaT':       ('loader', 'SWaT'),
        'WADI':       ('loader', 'WADI'),
        'Exathlon-1': ('tsb', 'Exathlon', 1),
        'Exathlon-2': ('tsb', 'Exathlon', 2),
    }

    for name, (src, *args) in at_datasets.items():
        print(f'\n--- {name} [AT] ---')
        try:
            if src == 'tsb':
                X, Y, _n = load_tsb_ad(args[0], args[1])
            else:
                X, Y, _n = load_dataset(args[0])
            print(f'  X={X.shape}, ratio={Y.mean():.4f}')
        except Exception as e:
            print(f'  加载失败: {e}')
            continue
        ps = run_method(baseline_anomaly_transformer, X, Y, SEEDS)
        if ps:
            results[f'AT_{name}'] = {'per_seed': ps, 'aggregate': aggregate(ps)}

    # ============ 保存 ============
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'extra_experiments.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n保存到 {out_path}')

    # 打印汇总
    print('\n' + '='*60)
    print('汇总')
    print('='*60)
    for name, methods in results.items():
        if 'aggregate' in methods:
            agg = methods['aggregate']
            print(f'{name:14s} AUPRC={agg.get("AUPRC","?")}  EF1={agg.get("EventF1_one2one","?")}')
        else:
            for mname, mdata in methods.items():
                agg = mdata.get('aggregate', {})
                print(f'{name:14s} {mname:5s} AUPRC={agg.get("AUPRC","?")}  EF1={agg.get("EventF1_one2one","?")}')


if __name__ == '__main__':
    main()
