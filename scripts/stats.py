#!/usr/bin/env python3
"""KHUNG thống kê theo PREREG.md (một scorer duy nhất từ logits). Chưa dùng cho kết quả; kiểm thử bằng dữ liệu giả.

score(logits, labels)            top1, top5, macro-F1, NLL, ECE (15 bin) -- tính từ logits đã lưu
welch(a, b)                      Welch hai phía (n=3 mỗi nhóm), mean/SD ddof=1 từ điểm chưa làm tròn
holm(pvals)                      hiệu chỉnh Holm trong một family
paired_bootstrap(la, lb, y, B)   hiệu top-1 (A - B) trên cùng ảnh test, CI 95% (chỉ bất định do lấy mẫu test)
"""
from __future__ import annotations

import numpy as np


def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def score(logits: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n, k = logits.shape
    lp = _log_softmax(logits)
    pred = lp.argmax(1)
    top5 = np.argsort(-lp, 1)[:, :5]
    conf = np.exp(lp.max(1))
    correct = pred == y
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    f1 = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"top1": float(correct.mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
            "macro_f1": float(np.mean(f1)), "nll": float(-lp[np.arange(n), y].mean()), "ece": float(ece)}


def welch(a, b) -> dict:
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    return {"mean_a": a.mean(), "sd_a": a.std(ddof=1), "mean_b": b.mean(), "sd_b": b.std(ddof=1),
            "diff": a.mean() - b.mean(), "t": float(t.statistic), "p": float(t.pvalue)}


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def paired_bootstrap(la: np.ndarray, lb: np.ndarray, y: np.ndarray, B: int = 10000, seed: int = 0) -> dict:
    ca = (np.asarray(la).argmax(1) == y).astype(float)
    cb = (np.asarray(lb).argmax(1) == y).astype(float)
    d = ca - cb
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "B": B}
