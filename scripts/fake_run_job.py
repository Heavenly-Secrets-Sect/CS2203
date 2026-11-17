#!/usr/bin/env python3
"""Thay thế food101.run_job để thử run_kaggle_food101.sh không cần GPU (FOOD101_RUN_CMD).
FAKE_FAIL=<chuỗi> làm job có id/model chứa chuỗi đó thoát mã 1. FAKE_SLEEP=<giây>."""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--regime"); ap.add_argument("--seed", type=int)
ap.add_argument("--out", required=True); ap.add_argument("--lr"); ap.add_argument("--epochs")
ap.add_argument("--train-frac"); ap.add_argument("--max-steps"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
print(f"fake job {a.model} {a.regime} seed={a.seed} lr={a.lr} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
if os.environ.get("FAKE_FAIL") and os.environ["FAKE_FAIL"] in a.model:
    sys.exit(1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"fake": True, "model": a.model}))
(out / "history.json").write_text("[]")
(out / "run_config.yaml").write_text(f"model: {a.model}\n")
(out / "test_predictions.jsonl").write_text('{"idx":0,"label":0,"pred":0}\n')
(out / "test_logits.npy").write_bytes(b"npy")
(out / "DONE").write_text("ok\n")
(out / "ckpt.pt").write_bytes(b"x" * 1000)   # phải KHÔNG vào zip
