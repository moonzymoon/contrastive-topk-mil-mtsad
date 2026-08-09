"""梯度归因验证 (创新点3): 用对比学习嵌入对输入的梯度做变量归因
替代 v1 失败的偏差向量归因, 用 SMD interpretation_label 验证."""
import sys, os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
from utils.repro import set_seed
from data.loaders import load_smd
from data.split import make_bags, bags_to_array, chronological_split
from models import CMHMIL


def load_interpretation(entity="machine-1-1"):
    path = os.path.join(PROJ, "datasets", "SMD", "interpretation_label", f"{entity}.txt")
    events = []
    if not os.path.exists(path): return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: continue
            rng, vars_str = line.split(':')
            if '-' not in rng: continue
            s, e = rng.split('-')
            vars_idx = [int(x) - 1 for x in vars_str.split(',') if x.strip().isdigit()]
            events.append((int(s), int(e), vars_idx))
    return events


def grad_attribution(model, x_instance):
    """对单个 instance 计算梯度归因.
    x_instance: (1, n, l, D) torch tensor
    返回: (D,) 每变量的归因分数."""
    x = x_instance.clone().requires_grad_(True)
    model.eval()
    scores, _, z = model(x, return_z=True)
    # 用异常分数对输入求梯度
    score = scores.max()
    score.backward()
    grad = x.grad.abs()  # (1, n, l, D)
    # 对 n 和 l 维求和 -> 每变量的总梯度贡献
    attribution = grad.sum(dim=(1, 2)).squeeze(0).cpu().numpy()  # (D,)
    return attribution


def evaluate_grad_attribution(entity="machine-1-1", k=3, seed=7):
    set_seed(seed)
    X, Y = load_smd(entity)
    T, D = X.shape
    events = load_interpretation(entity)
    if not events:
        print(f"  {entity}: 无interpretation_label"); return None
    c = int(T * 0.55)
    Xte, Yte = X[c:], Y[c:]
    bags = make_bags(Xte, Yte, L=128, l=16, stride=64)
    if not bags: return None
    Xb, yb = bags_to_array(bags)

    # 训练模型 (简化: 用已有 CMH-MIL)
    a = int(T * 0.35)
    wX, wY = bags_to_array(make_bags(X[:a], Y[:a], L=128, l=16, stride=64))
    model = CMHMIL(D, d_embed=24, k=3, channels=(32, 64), kernel_size=3, dropout=0.1, use_var_attn=True, pool='mean')
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    wXt = torch.from_numpy(wX); wYt = torch.from_numpy(wY.astype(np.float32))
    n_init = min(256, len(wXt))
    model.init_centers_kmeans(wXt[torch.randperm(len(wXt))[:n_init]], 'cpu')
    from losses.cmh_loss_v2 import CMHLossV2
    crit = CMHLossV2(w_contrast=0.5, w_topk=1.0)
    for ep in range(10):
        model.train(); perm = torch.randperm(len(wXt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            if len(idx) < 4: continue
            scores, _, z = model(wXt[idx], return_z=True)
            bag_z = z.reshape(len(idx), -1, z.shape[-1]).mean(1)
            loss, _ = crit(scores, wYt[idx], z.reshape(-1, z.shape[-1]), bag_z,
                           model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    # 对异常 bag 做梯度归因, 匹配真实标签
    hits_at_k = []
    hits_random = []
    anom_bags = [bi for bi, (insts, y) in enumerate(bags) if y == 1]
    rng = np.random.RandomState(seed)
    for bi in anom_bags:
        # bag 在 test 块内的局部坐标 (interpretation_label 是 test 局部偏移)
        bag_start_local = bi * 64
        bag_end_local = bag_start_local + 128
        real_vars = set()
        for (es, ee, vidx) in events:
            # interpretation_label 时间也是 test 局部, 直接和 bag 局部比较
            if es < bag_end_local and ee > bag_start_local:
                real_vars.update(vidx)
        if not real_vars: continue
        insts = bags[bi][0]  # (n, l, D)
        x_inst = torch.from_numpy(insts).unsqueeze(0).float()  # (1, n, l, D)
        try:
            attr = grad_attribution(model, x_inst)
            topk = set(np.argsort(-attr)[:k].tolist())
            hits_at_k.append(len(topk & real_vars) > 0)
        except: continue
        rand_topk = set(rng.choice(D, k, replace=False).tolist())
        hits_random.append(len(rand_topk & real_vars) > 0)
    hit_rate = float(np.mean(hits_at_k)) if hits_at_k else float('nan')
    random_rate = float(np.mean(hits_random)) if hits_random else float('nan')
    return {'entity': entity, 'Hit_at_3_grad': hit_rate, 'Hit_at_3_random': random_rate,
            'n_evaluated': len(hits_at_k)}


def main():
    results = {}
    label_dir = os.path.join(PROJ, "datasets", "SMD", "interpretation_label")
    entities = sorted([f.replace('.txt', '') for f in os.listdir(label_dir) if f.endswith('.txt')])[:8]
    print(f"评估 {len(entities)} 个实体的梯度归因")
    for ent in entities:
        try:
            r = evaluate_grad_attribution(ent, k=3, seed=7)
            if r:
                results[ent] = r
                print(f"  {ent}: Hit@3(grad)={r['Hit_at_3_grad']:.3f} vs random={r['Hit_at_3_random']:.3f} (n={r['n_evaluated']})")
        except Exception as e:
            print(f"  {ent} fail: {e}")
    if results:
        hits = [r['Hit_at_3_grad'] for r in results.values() if not np.isnan(r['Hit_at_3_grad'])]
        rands = [r['Hit_at_3_random'] for r in results.values() if not np.isnan(r['Hit_at_3_random'])]
        print(f"\n宏平均 (n={len(hits)}): Hit@3(grad)={np.mean(hits):.3f} vs random={np.mean(rands):.3f}")
        print(f"(v1偏差向量: 0.225 vs random 0.473 —— 失败)")
    out = os.path.join(PROJ, 'results', 'grad_attribution.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f"保存 {out}")


if __name__ == '__main__':
    main()
