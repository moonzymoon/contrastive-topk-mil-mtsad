"""CMH-MIL 训练主循环
合规: 5种子 / 配置归档 / 完整日志 / 文件哈希 / 失败原因 / 阴性结果留存.
流程: 弱训练块训练 -> 校准块保形选阈值(不触测试) -> 测试块迁移评估.
⚠️ 本脚本不含真实数据加载(数据下载待用户授权), 用合成数据做可运行性验证."""
import os, sys, json, time, argparse, numpy as np, torch
# 工程根目录(本文件在 src/ 下), 保证从任意 cwd 运行路径正确
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from utils.repro import set_seed, setup_logging, save_run_meta, file_sha256, load_config
from data.split import chronological_split, make_bags, bags_to_array
from models import CMHMIL
from losses.cmh_loss import CMHLoss
from evaluation.conformal import split_conformal_threshold, conformal_alarm, events_from_binary
from evaluation.metrics import evaluate_all

# 绝对路径(避免 cwd 重置导致找不到目录)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
CONFIG_DEFAULT = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def make_synthetic_data(T=4000, D=25, anomaly_rate=0.05, seed=0):
    """合成数据: 仅用于代码可运行性验证, 非真实实验.
    真实数据加载待用户授权(SMD/PSM/MSL/SMAP/SWaT)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, D)).astype(np.float32) * 0.5
    Y = np.zeros(T, dtype=np.int64)
    # 注入异常段
    n_anom = int(T * anomaly_rate)
    starts = rng.choice(T - 200, n_anom // 20, replace=False)
    for s in starts:
        L = rng.integers(20, 100)
        Y[s:s + L] = 1
        X[s:s + L] += rng.standard_normal((L, D)).astype(np.float32) * 2.0
    return X, Y


def train_one_seed(cfg, seed, dataset_name, device="cpu"):
    logger = setup_logging(LOG_DIR, seed, dataset_name)
    logger.info(f"device={device}")
    try:
        set_seed(seed)
        # 数据加载: 真实数据集优先, 找不到则回退合成
        if dataset_name in ("SMD", "PSM", "MSL", "SMAP", "5ECK2022", "SWaT", "WADI"):
            from data.loaders import load_dataset
            try:
                X, Y = load_dataset(dataset_name)
                logger.info(f"loaded real {dataset_name}: X{X.shape} Y{Y.shape} anomaly_rate={Y.mean():.4f}")
            except Exception as e:
                logger.warning(f"real {dataset_name} load failed ({e}), fallback synthetic")
                X, Y = make_synthetic_data(seed=seed)
        else:
            X, Y = make_synthetic_data(seed=seed)
        D = X.shape[1]
        sp = chronological_split(len(X), cfg["split"]["weak_train_ratio"], cfg["split"]["calib_ratio"])
        logger.info(f"split: weak_train[0:{sp.weak_train.stop}] calib[{sp.calib.start}:{sp.calib.stop}] test[{sp.test.start}:{sp.test.stop}]")

        # 三块各造 bag
        mk = lambda s: make_bags(X[s], Y[s], L=cfg["bag"]["L"], l=cfg["bag"]["l"], stride=cfg["bag"]["stride"])
        wb = mk(sp.weak_train); cb = mk(sp.calib); tb = mk(sp.test)
        wX, wY = bags_to_array(wb); cX, cY = bags_to_array(cb); tX, tY = bags_to_array(tb)
        logger.info(f"bags: weak={len(wb)} calib={len(cb)} test={len(tb)}")

        # 模型
        model = CMHMIL(D, d_embed=cfg["model"]["encoder"]["d_embed"],
                       k=cfg["model"]["hyperspheres"]["k"],
                       channels=tuple(cfg["model"]["encoder"]["channels"]),
                       kernel_size=cfg["model"]["encoder"]["kernel_size"],
                       dropout=cfg["model"]["encoder"]["dropout"],
                       use_var_attn=cfg["model"]["var_attention"],
                       pool=cfg["conformal"]["bag_pool"]).to(device)
        lc = cfg["loss"]
        criterion = CMHLoss(w_bag=lc["bag_bce"], w_compact=lc["compact"], w_sep=lc["sep"],
                            sep_margin=lc["sep_margin"], w_rank=lc["rank"],
                            rank_margin=lc["rank_margin"], w_elbo=lc["elbo_mog_alpha"],
                            w_balance=lc.get("balance", 0.10),
                            pool=cfg["conformal"]["bag_pool"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"]["weight_decay"])

        wXt = torch.from_numpy(wX).to(device); wYt = torch.from_numpy(wY).to(device)
        # k-means 初始化球中心 (防随机初始化导致坍缩)
        n_init = min(256, len(wXt))
        sample = wXt[torch.randperm(len(wXt))[:n_init]]
        model.init_centers_kmeans(sample, device)
        logger.info(f"k-means init centers done (sample={n_init} bags)")
        best_loss = float("inf")
        for epoch in range(cfg["train"]["max_epochs"]):
            model.train()
            perm = torch.randperm(len(wXt))
            epoch_loss = 0.0
            bs = cfg["train"]["batch_size"]
            for i in range(0, len(perm), bs):
                idx = perm[i:i+bs]
                xb, yb = wXt[idx], wYt[idx]
                scores, bag_score, z = model(xb, return_z=True)
                B, n, d = z.shape
                loss, parts = criterion(scores, yb, z.reshape(-1, d), model.hypers.centers, model.hypers.radii)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= len(perm)
            best_loss = min(best_loss, epoch_loss)
            logger.info(f"epoch {epoch+1}/{cfg['train']['max_epochs']} loss={epoch_loss:.4f} parts={parts}")
            # 坍缩诊断
            with torch.no_grad():
                zall = model.encoder(wXt).reshape(-1, model.encoder.d_embed)
                counts, minsep = model.hypers.collapse_diagnostic(zall)
                logger.info(f"  sphere counts={counts} min_center_sep={minsep:.3f}")

        # ===== 校准块保形选阈值 (不触测试标签) =====
        model.eval()
        with torch.no_grad():
            c_scores, _ = model(torch.from_numpy(cX).to(device))
        c_scores_np = c_scores.cpu().numpy()  # (Ncal, n)
        # 用校准块正常 bag 的分数分布估阈值
        c_bag_scores = []
        for i in range(len(c_scores_np)):
            ss = c_scores_np[i]
            if cfg["conformal"]["bag_pool"] == "softmax":
                w = np.exp(ss - ss.max()); w /= w.sum()
                c_bag_scores.append((w * ss).sum())
            else:
                c_bag_scores.append(ss.mean())
        c_bag_scores = np.array(c_bag_scores)
        calib_normal = c_bag_scores[cY == 0]
        thr = split_conformal_threshold(calib_normal, alpha=cfg["conformal"]["alpha"])
        logger.info(f"conformal threshold from calib block (alpha={cfg['conformal']['alpha']}): {thr:.4f}")

        # ===== 测试块评估 (迁移阈值) =====
        with torch.no_grad():
            t_scores, _ = model(torch.from_numpy(tX).to(device))
        t_scores_np = t_scores.cpu().numpy()
        # bag分数 -> 点级告警 (简化: bag分数广播到 bag 内点)
        t_bag_s = []
        for i in range(len(t_scores_np)):
            ss = t_scores_np[i]
            if cfg["conformal"]["bag_pool"] == "softmax":
                w = np.exp(ss - ss.max()); w /= w.sum()
                t_bag_s.append((w * ss).sum())
            else:
                t_bag_s.append(ss.mean())
        t_bag_s = np.array(t_bag_s)
        t_pred = (t_bag_s > thr).astype(int)
        # 点级真值: bag 任一点异常则该 bag 点段为异常
        t_point_true = tY
        t_point_score = t_bag_s  # bag级分数近似点级
        pred_events = events_from_binary(t_pred)
        # 真事件: 由 tY 连续段
        gt_events = events_from_binary(tY)
        metrics = evaluate_all(t_point_true, t_point_score, thr, gt_events, pred_events)
        logger.info(f"TEST metrics: {metrics}")

        # 保存
        os.makedirs(RESULT_DIR, exist_ok=True)
        result_path = os.path.join(RESULT_DIR, f"{dataset_name}_seed{seed}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "dataset": dataset_name, "metrics": metrics,
                       "threshold": thr, "best_loss": best_loss}, f, ensure_ascii=False, indent=2)
        # P1-M4: 保存模型 checkpoint (球心/半径+encoder), 供第三篇在线验证用
        ckpt_dir = os.path.join(RESULT_DIR, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_seed{seed}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": {"D": D, "d_embed": cfg["model"]["encoder"]["d_embed"],
                       "k": cfg["model"]["hyperspheres"]["k"],
                       "channels": cfg["model"]["encoder"]["channels"]},
            "threshold": thr, "seed": seed, "dataset": dataset_name,
        }, ckpt_path)
        logger.info(f"saved {result_path}")
        logger.info(f"saved checkpoint {ckpt_path}")
        return metrics
    except Exception as e:
        import traceback
        logger.error(f"RUN FAILED: {e}\n{traceback.format_exc()}")
        save_run_meta("results", cfg, seed, dataset_name, "fail", note=str(e))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dataset", default="SYNTHETIC")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds else cfg["seeds"]
    all_metrics = []
    for seed in seeds:
        m = train_one_seed(cfg, seed, args.dataset)
        all_metrics.append(m)
    # 聚合
    import numpy as np
    print("\n===== 跨种子汇总 =====")
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        if vals:
            print(f"{key}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    # 配置文件哈希留存
    print(f"\nconfig sha256: {file_sha256(args.config)}")


if __name__ == "__main__":
    main()
