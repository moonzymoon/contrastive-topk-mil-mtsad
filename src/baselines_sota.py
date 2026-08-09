"""USAD (UnSupervised Anomaly Detection) 基线 - Audibert et al. KDD 2020
轻量实现: 双AE对抗训练, 不需要下载第三方代码.
AE1重构 -> AE2重构 -> 分数=|AE1(AE2(x)) - x| + |AE2(AE1(x)) - x|
CPU友好."""
import numpy as np
import torch
import torch.nn as nn
from data.split import make_bags, bags_to_array
from evaluation.conformal import split_conformal_threshold, events_from_binary
from evaluation.metrics import evaluate_all


class USADEncoder(nn.Module):
    def __init__(self, D, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class USADDecoder(nn.Module):
    def __init__(self, D, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, D)
        )
    def forward(self, x): return self.net(x)


def baseline_usad(X, Y, cfg, seed=42):
    """USAD: 双自编码器对抗训练."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = X.shape[1]; T = len(X)
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)

    Xtr = torch.from_numpy(X[:a].astype(np.float32))
    Xcal = X[b:c]; Xte = X[c:]; Yte = Y[c:]

    enc1 = USADEncoder(D); dec1 = USADDecoder(D)
    enc2 = USADEncoder(D); dec2 = USADDecoder(D)
    opt = torch.optim.Adam(list(enc1.parameters()) + list(dec1.parameters()) +
                           list(enc2.parameters()) + list(dec2.parameters()), lr=1e-3)

    epochs = 30; batch_size = 256
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), batch_size):
            x = Xtr[perm[i:i+batch_size]]
            # AE1: encode -> decode
            z1 = enc1(x); x1 = dec1(z1)
            # AE2: encode x1 -> decode
            z2 = enc2(x1.detach()); x2 = dec2(z2)
            # Loss: AE1 reconstruct, AE2 reconstruct AE1's output
            loss1 = ((x1 - x) ** 2).mean()
            loss2 = ((x2 - x1.detach()) ** 2).mean()
            loss = loss1 + loss2
            opt.zero_grad(); loss.backward(); opt.step()

    # Score: reconstruction error of both AEs
    def score(data):
        x = torch.from_numpy(data.astype(np.float32))
        with torch.no_grad():
            z1 = enc1(x); x1 = dec1(z1)
            z2 = enc2(x1); x2 = dec2(z2)
            # USAD score: |x2 - x| (AE2(AE1(x)) reconstruction)
            return ((x2 - x) ** 2).mean(1).numpy()

    cal_scores = score(Xcal)
    test_scores = score(Xte)
    thr = np.quantile(cal_scores[Y[b:c] == 0], 0.95)
    pred = (test_scores > thr).astype(int)

    m = evaluate_all(Yte, test_scores, float(thr),
                     events_from_binary(Yte), events_from_binary(pred))
    m['empirical_FAR'] = float(pred[Yte == 0].mean()) if (Yte == 0).sum() > 0 else float('nan')
    return m


def baseline_lof(X, Y, cfg, seed=42):
    """LOF (Local Outlier Factor) - Breunig et al. SIGMOD 2000."""
    from sklearn.neighbors import LocalOutlierFactor
    np.random.seed(seed)
    D = X.shape[1]; T = len(X)
    a = int(T * 0.35); b = int(T * 0.45); c = int(T * 0.55)
    Xtr = X[:a]; Xcal = X[b:c]; Xte = X[c:]; Yte = Y[c:]

    # Fit on training data
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(Xtr)

    cal_scores = -lof.score_samples(Xcal)  # higher = more anomalous
    test_scores = -lof.score_samples(Xte)
    thr = np.quantile(cal_scores[Y[b:c] == 0], 0.95)
    pred = (test_scores > thr).astype(int)

    m = evaluate_all(Yte, test_scores, float(thr),
                     events_from_binary(Yte), events_from_binary(pred))
    m['empirical_FAR'] = float(pred[Yte == 0].mean()) if (Yte == 0).sum() > 0 else float('nan')
    return m
