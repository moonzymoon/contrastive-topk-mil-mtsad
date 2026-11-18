"""梯度归因 v3 版 (G-R11 修复, 2026-11-18)
==========================================================
旧版病灶 (三外审):
  - interpretation_label 时间偏移未处理 (旧坐标直接匹配 0.55T 切分)
  - 旧 make_bags (n=2, 32/128覆盖) + CMHLossV2 + pool='mean' + use_var_attn=True(死代码)
修复:
  - 轨1 official_split + Scaler + make_bags_v2 (n=8全覆盖)
  - v3 模型 (quantile半径/max pooling/无var_attn) + CMHLossV3 训练 (与主实验同管线)
  - interpretation 事件坐标平移: 全流坐标 = n_official_train + 事件偏移 (官方test内索引)
  - 每个 train_anom_bags=0 的实体 (官方train全正常), 与论文口径一致
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from data.protocol import official_split, Scaler, make_bags_v2
from data.loaders import load_smd
from models import CMHMIL
from losses.cmh_loss_v3 import CMHLossV3
from utils.repro import set_seed

L, l, BAG_STRIDE = 128, 16, 64
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "grad_attribution_c.json")


def load_interpretation(entity):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "SMD", "interpretation_label", f"{entity}.txt")
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
            vars_idx = [int(x) - 1 for x in vars_str.split(',') if x.strip().isdigit()]
            events.append((int(s), int(e), vars_idx))
    return events


def train_v3(X, Y, n_off, seed):
    """与 run_v3.run_v3 相同的轨1训练 (quantile/max/CMHLossV3), 返回模型与scaler."""
    from utils.repro import load_config
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    set_seed(seed)
    sp = official_split(len(Y), n_off)
    scaler = Scaler(X[sp["train"][0]:sp["train"][1]])
    Xs = scaler.transform(X)
    D = X.shape[1]
    model = CMHMIL(D, d_embed=24, k=3, channels=(32, 64), kernel_size=3,
                   dropout=0.1, use_var_attn=False, pool="max",
                   radius_mode="quantile", radius_q=0.9)
    crit = CMHLossV3(radius_mode="quantile")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    s, e = sp["train"]
    insts, y_bag, _, _ = make_bags_v2(Xs[s:e], Y[s:e], L, l, BAG_STRIDE)
    wX = torch.from_numpy(insts)
    wY = torch.from_numpy(y_bag.astype(np.float32))
    model.init_centers_kmeans(wX[torch.randperm(len(wX))[:256]])
    bs = cfg["train"]["batch_size"]
    model.train()
    for ep in range(cfg["train"]["max_epochs"]):
        perm = torch.randperm(len(wX))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            scores, bag_score, z = model(wX[idx], return_z=True)
            bag_z = z.mean(dim=1)
            loss, _ = crit(bag_score, wY[idx], scores, z.reshape(-1, z.shape[-1]),
                           bag_z, model.hypers.centers, model.hypers.radii)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            model.hypers.update_radii(z.reshape(-1, z.shape[-1]).detach())
    model.eval()
    return model, scaler, sp


def main(entities=("machine-1-1", "machine-1-2", "machine-2-1", "machine-3-1",
                   "machine-3-7", "machine-3-10"),
         seeds=(7,), k=3):
    results = {}
    for ent in entities:
        ent_res = {}
        for seed in seeds:
            set_seed(seed)
            X, Y, n_off = load_smd(ent)
            events = load_interpretation(ent)
            if not events:
                ent_res[seed] = {"note": "no interpretation labels"}
                continue
            model, scaler, sp = train_v3(X, Y, n_off, seed)
            Xs = scaler.transform(X)
            t0, t1 = sp["test"]

            # 事件坐标平移: 官方test内偏移 → 全流坐标, 再取与test段交集
            shifted = [(n_off + s, n_off + e, vars_idx) for s, e, vars_idx in events
                       if n_off + e > t0]  # 至少尾部落在test段
            if not shifted:
                ent_res[seed] = {"note": "no interpretation events inside test block"}
                continue

            hits, total = 0, 0
            D = X.shape[1]
            with torch.enable_grad():
                bs = 0
                while t0 + bs + L <= t1:
                    seg = Xs[t0 + bs:t0 + bs + L]
                    y_seg = Y[t0 + bs:t0 + bs + L]
                    insts = np.stack([seg[o:o + l] for o in range(0, L, l)])[None]
                    y_bag = int(y_seg.max() > 0)
                    if y_bag == 1:
                        xb = torch.from_numpy(insts).requires_grad_(True)
                        scores, _ = model(xb)
                        scores.max().backward()
                        attr = xb.grad.abs().sum(dim=(1, 2)).squeeze(0).numpy()  # (D,)
                        top3 = set(np.argsort(attr)[-k:].tolist())
                        # 找覆盖该bag的interpretation事件
                        b0, b1 = t0 + bs, t0 + bs + L
                        true_vars = set()
                        for es, ee, vv in shifted:
                            if es < b1 and ee > b0:  # 与bag有交集
                                true_vars.update(vv)
                        if true_vars:
                            total += 1
                            if top3 & true_vars:
                                hits += 1
                    bs += BAG_STRIDE
            hit3 = hits / total if total else float("nan")
            ent_res[seed] = {"hit3": hit3, "n_anom_bags": total}
            print(f"[{ent} s{seed}] Hit@3={hit3:.3f} (n={total})")
        results[ent] = ent_res

    # 汇总
    all_hits = [r["hit3"] for r in results.values() for r2 in
                ([r] if isinstance(r, dict) and "hit3" in r else [])
                for r in [r2]]
    flat = []
    for ent, er in results.items():
        for sd, r in er.items():
            if isinstance(r, dict) and "hit3" in r:
                flat.append(r["hit3"])
    if flat:
        results["_aggregate"] = {"hit3_mean": statistics.mean(flat),
                                 "n_entities": len(results) - 1,
                                 "note": "random baseline D-dependent; "
                                         "see paper for per-entity D"}
        print(f"\n宏平均 Hit@3 = {statistics.mean(flat):.3f} "
              f"(旧版0.438, random≈0.566口径见论文)")
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
