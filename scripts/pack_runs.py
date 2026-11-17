#!/usr/bin/env python3
"""Đóng gói kết quả của session thành MỘT zip trong CI_OUTPUT_DIR, ghi nguyên tử (tmp + os.replace).

Bao gồm: outputs/**/{metrics.json,history.json,run_config.yaml,test_predictions.jsonl,*.npy},
logs/*.log, job_times.csv, ci_env.json. KHÔNG bao gồm checkpoint (*.pt, *.pth, *.ckpt, *.safetensors, *.bin)
hay cache. Gọi sau MỖI job: session bị cắt chỉ mất job đang chạy.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

KEEP_EXT = {".json", ".jsonl", ".yaml", ".yml", ".csv", ".log", ".npy", ".txt"}
DENY_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl"}
DENY_DIR = {"ckpt", "checkpoints", "cache", "hf", "torch", "__pycache__"}


def wanted(rel: Path) -> bool:
    if rel.suffix.lower() in DENY_EXT or any(p in DENY_DIR for p in rel.parts[:-1]):
        return False
    return rel.suffix.lower() in KEEP_EXT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session", type=int, required=True)
    a = ap.parse_args(argv)
    persist, out_dir = Path(a.persist), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"food101_s{a.session}.zip"
    tmp = out_dir / f".food101_s{a.session}.zip.tmp"
    n = 0
    with zipfile.ZipFile(tmp, "w") as z:
        for top in ("outputs", "logs"):
            for p in sorted((persist / top).rglob("*")) if (persist / top).exists() else []:
                if p.is_file():
                    rel = p.relative_to(persist)
                    if wanted(rel):
                        comp = zipfile.ZIP_STORED if p.suffix == ".npy" else zipfile.ZIP_DEFLATED
                        z.write(p, rel.as_posix(), compress_type=comp)
                        n += 1
        for name in ("job_times.csv", "ci_env.json"):
            p = persist / name
            if p.is_file():
                z.write(p, name, compress_type=zipfile.ZIP_DEFLATED)
                n += 1
    os.replace(tmp, final)
    print(f"pack: {final} ({n} file, {final.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
