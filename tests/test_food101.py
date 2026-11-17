import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from food101 import evaluate as E
from food101 import splits as S
from food101.run_job import main as run_main

ROOT = Path(__file__).resolve().parents[1]


def run_smoke(out, regime="ft", model="vit_tiny", seed=1, *extra):
    run_main(["--model", model, "--regime", regime, "--seed", str(seed), "--out", str(out), "--smoke", *extra])
    return Path(out)


# ---- scorer
def test_scorer_hand_computed():
    logits = np.log(np.array([[0.7, 0.2, 0.1],     # pred 0 gold 0
                              [0.1, 0.8, 0.1],     # pred 1 gold 2
                              [0.3, 0.3, 0.4],     # pred 2 gold 2
                              [0.5, 0.4, 0.1]]))   # pred 0 gold 1
    gold = np.array([0, 2, 2, 1])
    m = E.score(logits, gold, n_bins=15)
    assert m["top1"] == 0.5
    assert m["top5"] == 1.0                         # 3 lop -> top5 = tat ca
    # F1: lop0 tp1 fp1 fn0 -> 2/3; lop1 tp0 -> 0; lop2 tp1 fp0 fn1 -> 2/3
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)
    nll = -(np.log(0.7) + np.log(0.1) + np.log(0.4) + np.log(0.4)) / 4
    assert m["nll"] == pytest.approx(nll)
    # ECE: conf .7 (ok), .8 (sai), .4 (ok), .5 (sai) -> moi anh 1 bin rieng
    expect = (abs(1 - .7) + abs(0 - .8) + abs(1 - .4) + abs(0 - .5)) / 4
    assert m["ece"] == pytest.approx(expect)
    idx, prob = E.top5(logits)
    assert idx.shape == (4, 3) and idx[0, 0] == 0 and prob[0, 0] == pytest.approx(0.7)


# ---- split
def test_split_deterministic_stratified():
    labels = np.repeat(np.arange(5), 40)
    t1, d1 = S.make_dev_split(labels, 0.1, 1234)
    t2, d2 = S.make_dev_split(labels, 0.1, 1234)
    assert (d1 == d2).all() and S.ids_hash(d1) == S.ids_hash(d2)
    assert len(d1) == 20 and (np.bincount(labels[d1]) == 4).all()
    assert not set(t1) & set(d1) and len(t1) + len(d1) == 200
    _, d3 = S.make_dev_split(labels, 0.1, 99)
    assert (d1 != d3).any()
    sub = S.subsample_stratified(t1, labels, 0.2)
    assert (sub == S.subsample_stratified(t1, labels, 0.2)).all()


# ---- schema + regimes
@pytest.fixture(scope="module")
def ft_dir(tmp_path_factory):
    return run_smoke(tmp_path_factory.mktemp("ft") / "r")


def test_schema(ft_dir):
    for f in ["test_predictions.jsonl", "test_logits.npy", "dev_logits.npy", "metrics.json",
              "history.json", "run_config.yaml"]:
        assert (ft_dir / f).exists(), f
    assert not (ft_dir / "checkpoint.pt").exists()
    tl = np.load(ft_dir / "test_logits.npy")
    assert tl.dtype == np.float16 and tl.shape == (40, 10)
    rows = [json.loads(l) for l in (ft_dir / "test_predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and set(rows[0]) == {"id", "gold", "pred", "top5_idx", "top5_prob"}
    m = json.loads((ft_dir / "metrics.json").read_text())
    assert {"top1", "top5", "macro_f1", "nll", "ece"} <= set(m["test"])
    # metrics tinh lai tu logits da luu
    gold = np.array([r["gold"] for r in rows])
    again = E.score(tl.astype(np.float32), gold)
    assert again["top1"] == m["test"]["top1"] and again["nll"] == pytest.approx(m["test"]["nll"])
    assert np.mean([r["pred"] == r["gold"] for r in rows]) == m["test"]["top1"]
    rc = yaml.safe_load((ft_dir / "run_config.yaml").read_text())
    for k in ["command", "git_commit", "versions", "gpu", "model", "hyperparameters", "seed",
              "data", "time", "peak_memory"]:
        assert k in rc
    assert rc["data"]["dev_ids_hash"] and rc["hyperparameters"]["effective_batch"] == 32  # smoke
    h = json.loads((ft_dir / "history.json").read_text())
    assert len(h["finetune"]["epochs"]) == 5 and "dev_acc" in h["finetune"]["epochs"][0]


def test_same_seed_same_result(tmp_path, ft_dir):
    d2 = run_smoke(tmp_path / "again")
    assert np.array_equal(np.load(ft_dir / "test_logits.npy"), np.load(d2 / "test_logits.npy"))
    d3 = run_smoke(tmp_path / "other", "ft", "vit_tiny", 2)
    assert not np.array_equal(np.load(ft_dir / "test_logits.npy"), np.load(d3 / "test_logits.npy"))


def test_lp_lpft(tmp_path):
    lp = run_smoke(tmp_path / "lp", "lp")
    h = json.loads((lp / "history.json").read_text())
    assert "finetune" not in h and len(h["lp"]["grid"]) == 5
    lpft = run_smoke(tmp_path / "lpft", "lpft")
    h = json.loads((lpft / "history.json").read_text())
    assert "lp" in h and "finetune" in h
    # head khoi tao tu LP: dev acc LP >= ngau nhien (10 lop, 20 anh dev)
    assert json.loads((lp / "metrics.json").read_text())["dev"]["top1"] > 0.1


def test_sweep_and_zeroshot(tmp_path):
    sw = run_smoke(tmp_path / "sw", "sweep", "vit_tiny", 42, "--lr", "1e-4")
    m = json.loads((sw / "metrics.json").read_text())
    assert "dev" in m and "test" not in m
    assert not (sw / "test_logits.npy").exists() and (sw / "dev_logits.npy").exists()
    rc = yaml.safe_load((sw / "run_config.yaml").read_text())
    assert rc["hyperparameters"]["epochs"] == 2 and rc["hyperparameters"]["train_frac"] == 0.2
    zs = run_smoke(tmp_path / "zs", "zeroshot", "clip_base")
    assert (zs / "test_logits.npy").exists()
    assert yaml.safe_load((zs / "run_config.yaml").read_text())["model"]["zeroshot_impl"] == "SMOKE-STUB"


def test_defaults_unchanged_and_flags(tmp_path, ft_dir):
    # --save-ckpt chi them checkpoint, logits giu nguyen
    d = run_smoke(tmp_path / "ck", "ft", "vit_tiny", 1, "--save-ckpt")
    assert (d / "checkpoint.pt").exists()
    assert np.array_equal(np.load(d / "test_logits.npy"), np.load(ft_dir / "test_logits.npy"))
    rc = yaml.safe_load((ft_dir / "run_config.yaml").read_text())
    assert rc["hyperparameters"]["epochs"] == 5 and rc["hyperparameters"]["weight_decay"] == 0.05
    assert rc["hyperparameters"]["label_smoothing"] == 0.1


def test_cli_exit_code_and_stdout(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    ok = subprocess.run([sys.executable, "-m", "food101.run_job", "--model", "vit_tiny", "--regime", "ft",
                         "--seed", "1", "--out", str(tmp_path / "o"), "--smoke", "--epochs", "1"],
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0
    assert json.loads(ok.stdout.strip().splitlines()[-1]) == {"status": "ok", "dir": str(tmp_path / "o")}
    bad = subprocess.run([sys.executable, "-m", "food101.run_job", "--model", "nope", "--regime", "ft",
                          "--seed", "1", "--out", str(tmp_path / "b"), "--smoke"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode != 0
    assert json.loads(bad.stdout.strip().splitlines()[-1])["status"] == "error"
    nolr = subprocess.run([sys.executable, "-m", "food101.run_job", "--model", "vit_tiny", "--regime", "sweep",
                           "--seed", "1", "--out", str(tmp_path / "c")], capture_output=True, text=True, env=env)
    assert nolr.returncode != 0


def test_models_yaml_complete():
    m = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    assert set(m) == {"vit_tiny", "vit_small", "vit_base", "vit_large", "convnext_base", "resnet50",
                      "deit_base", "dinov2_base", "clip_base"}
    for k, v in m.items():
        assert {"timm_id", "mean", "std", "interpolation", "img_size", "micro_batch", "lr"} <= set(v), k
        assert v["lr"] is None
