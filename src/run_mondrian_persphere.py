"""H2实验: Mondrian per-sphere (regime-conditional) FAR 验证 (Eq.6c)
在SMAP(球分布最均衡)上, 对每个球单独算 conformal 阈值, 报告每球经验FAR.
验证: per-sphere FAR是否在 within-regime 下满足 alpha (Eq.6c)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "src"))
from utils.repro import load_config, set_seed
from data.loaders import load_dataset
from data.split import chronological_split, make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss import CMHLoss
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all

cfg = load_config(os.path.join(PROJ, 'configs', 'default.yaml'))
SEEDS = [7, 42, 123, 202, 999]
ALPHA = 0.05


def run_mondrian(dataset_name, seed):
    """跑 per-sphere Mondrian: 训练模型 -> 每球单独阈值 -> 每球经验FAR."""
    set_seed(seed)
    X, Y = load_dataset(dataset_name)
    D = X.shape[1]
    T = len(X)
    # 四段切分
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)
    mk = lambda s, e: make_bags(X[s:e], Y[s:e], L=cfg['bag']['L'], l=cfg['bag']['l'], stride=cfg['bag']['stride'])
    wX, wY = bags_to_array(mk(0, a))
    cX, cY = bags_to_array(mk(b, c))
    tX, tY = bags_to_array(mk(c, T))

    k = 3
    model = CMHMIL(D, d_embed=cfg['model']['encoder']['d_embed'], k=k,
                   channels=tuple(cfg['model']['encoder']['channels']),
                   kernel_size=cfg['model']['encoder']['kernel_size'],
                   dropout=cfg['model']['encoder']['dropout'],
                   use_var_attn=cfg['model']['var_attention'], pool=cfg['conformal']['bag_pool'])
    lc = cfg['loss']
    crit = CMHLoss(w_bag=lc['bag_bce'], w_compact=lc['compact'], w_sep=lc['sep'], sep_margin=lc['sep_margin'],
                   w_rank=lc['rank'], rank_margin=lc['rank_margin'], w_elbo=lc['elbo_mog_alpha'],
                   w_balance=lc.get('balance', 0.10))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    wXt = torch.from_numpy(wX); wYt = torch.from_numpy(wY.astype(np.float32))
    n_init = min(256, len(wXt))
    model.init_centers_kmeans(wXt[torch.randperm(len(wXt))[:n_init]], 'cpu')
    bs = cfg['train']['batch_size']
    for ep in range(cfg['train']['max_epochs']):
        model.train(); perm = torch.randperm(len(wXt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]; xb, yb = wXt[idx], wYt[idx]
            scores, _, z = model(xb, return_z=True)
            loss, _ = crit(scores, yb, z.reshape(-1, z.shape[-1]), model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip']); opt.step()
    model.eval()

    # 取 instance 级分数 + 球分配
    with torch.no_grad():
        # CMHMIL forward: 返回 (scores, info, z)
        c_scores, c_info, c_z = model(torch.from_numpy(cX), return_z=True)
        t_scores, t_info, t_z = model(torch.from_numpy(tX), return_z=True)
    c_z_np = c_z.reshape(-1, c_z.shape[-1]).cpu().numpy()  # (N_calib*n, d)
    t_z_np = t_z.reshape(-1, t_z.shape[-1]).cpu().numpy()
    centers = model.hypers.centers.detach().cpu().numpy()  # (k, d)

    # 每个 instance 分配到最近球
    def assign(z):
        dist = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)  # (N, k)
        return np.argmin(dist, axis=1)  # (N,)

    # instance 级分数: 用 bag 分数展开到 instance (简化: 用 bag score 填充)
    # 更准确: 用 raw distance score
    def inst_score(z):
        dist = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)  # (N,k)
        return dist.min(axis=1)  # 到最近球距离作为分数

    c_inst = inst_score(c_z_np)
    t_inst = inst_score(t_z_np)
    c_assign = assign(c_z_np)
    t_assign = assign(t_z_np)

    # 每球 conformal 阈值 (calib 的 normal instance)
    # bag 标签 -> instance 标签: bag normal 则其所有 instance normal
    c_bag_norm = np.repeat(cY == 0, c_z.shape[1])[:len(c_inst)]
    t_bag_norm = np.repeat(tY == 0, t_z.shape[1])[:len(t_inst)]

    per_sphere_far = {}
    global_far = None
    for j in range(k):
        mask_c = (c_assign == j) & c_bag_norm
        mask_t = (t_assign == j) & t_bag_norm
        n_j = mask_c.sum()
        if n_j < 30:
            per_sphere_far[f'sphere{j}'] = {'n_calib': int(n_j), 'FAR': None, 'note': 'too few (<30), fallback'}
            continue
        calib_j = c_inst[mask_c]
        tau_j = split_conformal_threshold(calib_j, alpha=ALPHA)
        test_j = t_inst[mask_t]
        if len(test_j) == 0:
            per_sphere_far[f'sphere{j}'] = {'n_calib': int(n_j), 'n_test': 0, 'FAR': None, 'note': 'no normal test'}
            continue
        far_j = float(np.mean(test_j > tau_j))
        per_sphere_far[f'sphere{j}'] = {'n_calib': int(n_j), 'n_test': int(len(test_j)), 'tau': float(tau_j), 'FAR': far_j}

    # 全局对照
    all_calib_norm = c_inst[c_bag_norm]
    tau_global = split_conformal_threshold(all_calib_norm, alpha=ALPHA)
    global_far = float(np.mean(t_inst[t_bag_norm] > tau_global))

    return {'dataset': dataset_name, 'seed': seed, 'per_sphere': per_sphere_far,
            'global_FAR': global_far, 'alpha': ALPHA}


results = {}
for ds in ['SMAP', 'MSL']:
    print(f"\n=== {ds} Mondrian per-sphere ===")
    per_seed = []
    for seed in SEEDS:
        try:
            r = run_mondrian(ds, seed)
            per_seed.append(r)
            ps = {k: v.get('FAR') for k, v in r['per_sphere'].items()}
            print(f"  seed{seed}: global_FAR={r['global_FAR']:.4f} per_sphere_FAR={ps}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  seed{seed} fail: {e}")
    results[ds] = per_seed

json.dump(results, open(os.path.join(PROJ, 'results', 'mondrian_persphere.json'), 'w'), indent=2)
print("\n保存 results/mondrian_persphere.json")
PYEOF