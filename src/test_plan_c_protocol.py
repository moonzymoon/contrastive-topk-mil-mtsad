"""步骤1d 单测: 方案C 双轨协议 (任务书三条断言 + annot 块验证).
铁律4: 附数值断言; 铁律5: 任务书预设 28517 与实测 28479 不符 → 以数据文件为准 (28479),
已在执行汇报中记录 (任务书笔误, wc -l 实测 28479 行).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

from data.protocol import official_split, weak_annotation_split, make_bags_v2
from data.loaders import load_dataset

print("=== 断言1: SMD n_official_train = 28479 (wc -l 实测; 任务书 28517 系笔误) ===")
X, Y, n = load_dataset("SMD")
assert n == 28479, f"SMD n_official_train={n}, expect 28479"
assert len(X) == 28479 * 2, f"SMD 全流长 {len(X)} 应为 2*28479"
print(f"  SMD: n_official_train={n}, T={len(X)} ✓")

print("\n=== 断言2: 轨1各块不越界 + 顺序 + 覆盖 ===")
sp = official_split(len(X), n)
seq = [sp["train"], sp["val"], sp["calib"], sp["test"]]
for (s1, e1), (s2, e2) in zip(seq, seq[1:]):
    assert e1 == s2, f"块不连续: {e1} != {s2}"
assert seq[0][0] == 0 and seq[-1][1] == len(X), "首尾越界"
a, b, c = sp["train"][1], sp["val"][1], sp["calib"][1]
assert a == round(28479 * 0.70) and b == round(28479 * 0.85) and c == 28479
print(f"  train={sp['train']} val={sp['val']} calib={sp['calib']} test={sp['test']} ✓")
print(f"  mode={sp['mode']}, 各块点数: {[e-s for s,e in seq]}")

print("\n=== 断言3: SMAP train块点数 = 官方train*0.70 (降采样口径) ===")
Xs, Ys, ns = load_dataset("SMAP")
sps = official_split(len(Ys), ns)
assert sps["mode"] == "official"
train_len = sps["train"][1] - sps["train"][0]
expect = round(ns * 0.70)
assert train_len == expect, f"SMAP train_len={train_len}, expect {expect}"
print(f"  SMAP: n_official_train(降采样后)={ns}, train_len={train_len} = 0.70*{ns} ✓")

print("\n=== 断言4: SMAP 轨1 train_anom_bags = 0 (官方train全正常) ===")
insts, y_bag, starts, rel = make_bags_v2(
    Xs[sps["train"][0]:sps["train"][1]], Ys[sps["train"][0]:sps["train"][1]],
    L=128, l=16, bag_stride=64)
n_anom = int(y_bag.sum())
assert n_anom == 0, f"SMAP 轨1 train_anom_bags={n_anom}, 应为0 (官方train无异常)"
print(f"  train_bags={len(y_bag)}, anom_bags={n_anom} ✓")

print("\n=== 断言5: TSB 整流回退 (Exathlon n=None → single-stream 35/10/10/45) ===")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.tsb_ad_loader import load_tsb_ad
Xe, Ye, ne = load_tsb_ad("Exathlon", 1)
assert ne is None
spe = official_split(len(Ye), ne)
assert spe["mode"] == "single-stream"
assert spe["train"][1] == round(len(Ye) * 0.35)
print(f"  Exathlon-1: mode={spe['mode']}, train_end={spe['train'][1]} = 0.35*T ✓")

print("\n=== 断言6: 轨2 annot 块 (官方test前15%) ===")
sp2 = weak_annotation_split(len(Ys), ns)
t0, t1 = official_split(len(Ys), ns)["test"]
n_annot = round((t1 - t0) * 0.15)
assert sp2["annot"] == (t0, t0 + n_annot), sp2["annot"]
assert sp2["test"] == (t0 + n_annot, t1), sp2["test"]
assert sp2["train"] == sps["train"], "轨2不得改动轨1的train/val/calib"
# annot 块含异常 (官方test有异常, 前段应能提供双类bag)
insts2, yb2, _, _ = make_bags_v2(
    Xs[sp2["annot"][0]:sp2["annot"][1]], Ys[sp2["annot"][0]:sp2["annot"][1]],
    L=128, l=16, bag_stride=64)
print(f"  SMAP annot块: {sp2['annot']}, bags={len(yb2)}, anom_bags={int(yb2.sum())}")
print(f"  test段(85%): {sp2['test']} ✓")

print("\n=== 断言7: TSB 轨2 = single-stream+annot ===")
sp3 = weak_annotation_split(len(Ye), ne)
assert sp3["mode"] == "single-stream+annot" and sp3["annot"][0] == spe["test"][0]
print(f"  Exathlon-1 轨2: mode={sp3['mode']}, annot={sp3['annot']} ✓")

print("\n全部单测通过 — 步骤1完成")
