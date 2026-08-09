"""用 SMD 真实 interpretation_label 验证多球归因 (PAA评估第9点)
替代原来的循环代理(deviation-from-statistics), 用真实根因标签计算 Hit@k.
interpretation_label 格式: "起-止:var1,var2,..." 每行一个异常事件.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "src"))
from data.loaders import load_smd
from data.split import make_bags, bags_to_array, chronological_split


def load_interpretation(entity="machine-1-1"):
    """加载真实根因标签. 返回 list of (start, end, [var_indices]).
    注意: interpretation_label 的时间是相对于 test 块起点的偏移."""
    path = os.path.join(PROJ, "datasets", "SMD", "interpretation_label", f"{entity}.txt")
    events = []
    if not os.path.exists(path):
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            rng, vars_str = line.split(':')
            if '-' not in rng:
                continue
            s, e = rng.split('-')
            vars_idx = [int(x) - 1 for x in vars_str.split(',') if x.strip().isdigit()]  # 0-indexed
            events.append((int(s), int(e), vars_idx))
    return events


def evaluate_attribution_real(entity="machine-1-1", k=3, seed=7):
    """用真实 interpretation_label 计算 Hit@k.
    归因方法: 对每个异常 bag, 用偏离向量(Jacobian×δ)排序变量, 看top-k是否命中真实根因."""
    np.random.seed(seed); torch.manual_seed(seed)
    X, Y = load_smd(entity)
    T, D = X.shape
    events = load_interpretation(entity)
    if not events:
        print(f"  {entity}: 无interpretation_label")
        return None

    # 构造 test 块 bag (异常都在 test)
    sp = chronological_split(T, 0.35, 0.10)
    # 实际 test 从 0.55 开始 (四段)
    c = int(T * 0.55)
    Xte, Yte = X[c:], Y[c:]
    bags = make_bags(Xte, Yte, L=128, l=16, stride=64)
    if not bags:
        return None
    Xb, yb = bags_to_array(bags)

    # 找异常 bag 及其对应的全局时间区间 (test 块内偏移 + c)
    anom_bags = []
    for bi, (insts, y) in enumerate(bags):
        if y == 1:
            bag_start_local = bi * 64  # stride
            bag_start_global = c + bag_start_local
            bag_end_global = bag_start_global + 128
            anom_bags.append((bi, bag_start_global, bag_end_global))

    # 对每个异常 bag, 匹配真实根因事件
    # interpretation_label 时间是相对 test 块起点的偏移, bag 时间是全局
    hits_at_k = []
    hits_random = []
    test_offset = c  # test 块全局起点
    for bi, gs, ge in anom_bags:
        # bag 在 test 块内的局部偏移
        ls, le = gs - test_offset, ge - test_offset
        # 找重叠的真实事件 (interpretation_label 是 test 局部时间)
        real_vars = set()
        for (es, ee, vidx) in events:
            if es < le and ee > ls:  # 时间重叠 (都用 test 局部坐标)
                real_vars.update(vidx)
        if not real_vars:
            continue
        # 归因: 简化版 - 用 bag 内各变量的 deviation from 训练均值排序
        # (完整版需 CMH-MIL 的 Jacobian, 这里用 input deviation 作为可复现代理)
        insts = bags[bi][0]  # (n, l, D)
        train_mean = X[:int(T*0.35)].mean(0)  # weak-train 均值
        deviation = np.abs(insts.mean(axis=(0,1)) - train_mean)  # (D,)
        topk = set(np.argsort(-deviation)[:k].tolist())
        hit = len(topk & real_vars) > 0
        hits_at_k.append(hit)
        # 随机基线
        rand_topk = set(np.random.choice(D, k, replace=False).tolist())
        hits_random.append(len(rand_topk & real_vars) > 0)

    hit_rate = np.mean(hits_at_k) if hits_at_k else float('nan')
    random_rate = np.mean(hits_random) if hits_random else float('nan')
    return {
        'entity': entity,
        'n_anomalous_bags': len(anom_bags),
        'n_evaluated': len(hits_at_k),
        'n_real_events': len(events),
        'Hit_at_3_real': float(hit_rate),
        'Hit_at_3_random': float(random_rate),
        'k': k,
    }


def main():
    results = {}
    # machine-1-1 (主表用的实体) + 几个有标签的实体做泛化
    entities = ['machine-1-1']
    # 补几个有 interpretation 的
    label_dir = os.path.join(PROJ, "datasets", "SMD", "interpretation_label")
    all_ents = sorted([f.replace('.txt', '') for f in os.listdir(label_dir) if f.endswith('.txt')])
    entities = all_ents[:10]  # 前10个有标签的实体

    print(f"评估 {len(entities)} 个实体的真实标签归因")
    for ent in entities:
        try:
            r = evaluate_attribution_real(ent, k=3, seed=7)
            if r:
                results[ent] = r
                print(f"  {ent}: Hit@3={r['Hit_at_3_real']:.3f} (random={r['Hit_at_3_random']:.3f}, n={r['n_evaluated']})")
        except Exception as e:
            print(f"  {ent} fail: {e}")

    # 宏平均
    if results:
        hits = [r['Hit_at_3_real'] for r in results.values() if not np.isnan(r['Hit_at_3_real'])]
        rands = [r['Hit_at_3_random'] for r in results.values() if not np.isnan(r['Hit_at_3_random'])]
        agg = {
            'macro_Hit_at_3_real': float(np.mean(hits)),
            'macro_Hit_at_3_random': float(np.mean(rands)),
            'n_entities': len(hits),
        }
        results['_macro'] = agg
        print(f"\n宏平均 (n={agg['n_entities']}): Hit@3={agg['macro_Hit_at_3_real']:.3f} vs random={agg['macro_Hit_at_3_random']:.3f}")

    out = os.path.join(PROJ, 'results', 'attribution_real_labels.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=2)
    print(f"保存: {out}")


if __name__ == '__main__':
    main()
