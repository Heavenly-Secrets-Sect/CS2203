"""Scorer duy nhat: moi chi so tinh tu logits (da luu). top1, top5, macro-F1, NLL, ECE (15 bin)."""
import json
from pathlib import Path

import numpy as np


def _softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def top5(logits):
    """Tra ve (idx, prob) top-5; pha vong (tie) xu ly on dinh theo chi so lop."""
    logits = np.asarray(logits, dtype=np.float32)
    k = min(5, logits.shape[1])
    idx = np.argsort(-logits.astype(np.float64), axis=1, kind="stable")[:, :k]
    p = _softmax(logits)
    return idx, np.take_along_axis(p, idx, axis=1)


def macro_f1(pred, gold, num_classes):
    cm = np.bincount(gold * num_classes + pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)  # lop khong xuat hien: F1=0
    return float(f1.mean())


def ece(probs, gold, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == gold).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out, n = 0.0, len(gold)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if lo == 0.0:
            m = m | (conf == 0.0)
        if m.any():
            out += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def score(logits, gold, n_bins=15):
    logits = np.asarray(logits, dtype=np.float32)
    gold = np.asarray(gold, dtype=np.int64)
    n, c = logits.shape
    idx, _ = top5(logits)
    pred = idx[:, 0]
    probs = _softmax(logits)
    ls = _log_softmax(logits)
    per_class = [float((pred[gold == k] == k).mean()) if (gold == k).any() else None for k in range(c)]
    return {
        "n": int(n),
        "top1": float((pred == gold).mean()),
        "top5": float((idx == gold[:, None]).any(axis=1).mean()),
        "macro_f1": macro_f1(pred, gold, c),
        "nll": float(-ls[np.arange(n), gold].mean()),
        "ece": ece(probs, gold, n_bins),
        "ece_bins": n_bins,
        "per_class_acc": per_class,
    }


def write_predictions(path, ids, gold, logits):
    idx, prob = top5(logits)
    with open(path, "w") as f:
        for i, g, ti, tp in zip(ids, gold, idx, prob):
            f.write(json.dumps({"id": i, "gold": int(g), "pred": int(ti[0]),
                                "top5_idx": [int(x) for x in ti],
                                "top5_prob": [round(float(x), 6) for x in tp]}) + "\n")


def score_files(logits_path, labels, n_bins=15):
    """Tinh lai tu file .npy (float16) - dung chinh du lieu nguoi phan tich se doc."""
    return score(np.load(logits_path).astype(np.float32), labels, n_bins)


if __name__ == "__main__":  # python -m food101.evaluate logits.npy labels.npy
    import sys
    m = score_files(sys.argv[1], np.load(sys.argv[2]))
    m.pop("per_class_acc")
    print(json.dumps(m, indent=2))
