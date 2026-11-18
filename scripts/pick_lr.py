#!/usr/bin/env python3
"""Chọn LR theo DEV acc từ kết quả stage sweep (PREREG: không dùng test). Ghi configs/selected_lr.json.

  python3 scripts/pick_lr.py food101_r2_s1.zip food101_r5_s1.zip [...]   # round 2 + round 5 (sweep mở rộng) [--out configs/selected_lr.json]
LR tốt nhất ở rìa lưới => in cảnh báo: PREREG yêu cầu mở thêm một bậc cho model đó và ghi DEVIATIONS.md (cần plan mới).
Khoá dev acc trong metrics.json là giả định (DEV_KEYS); sửa nếu food101.run_job dùng tên khác.
"""
import argparse, json, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import ALL_MODELS, LR_TAG, LR_PATH, expected_lrs  # noqa: E402

DEV_KEYS = ("dev_acc", "dev_top1", "val_acc", "dev.acc", "dev.top1", "dev.accuracy", "dev_accuracy")


def dev_acc(metrics: dict):
    flat = {}
    def walk(d, p=""):
        for k, v in d.items():
            (walk(v, f"{p}{k}.") if isinstance(v, dict) else flat.__setitem__(f"{p}{k}", v))
    walk(metrics)
    for k in DEV_KEYS:
        if isinstance(flat.get(k), (int, float)):
            return float(flat[k])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(LR_PATH))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        roots = []
        for i, r in enumerate(a.runs):
            p = Path(r)
            if p.suffix == ".zip":
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if not str((Path(tmp) / str(i) / n).resolve()).startswith(str((Path(tmp) / str(i)).resolve())):
                            sys.exit("zip nguy hiểm")
                    z.extractall(Path(tmp) / str(i))
                p = Path(tmp) / str(i)
            roots.append(p / "outputs" if (p / "outputs").is_dir() else p)
        res = {}
        for m in ALL_MODELS:
            for lr, tag in LR_TAG.items():
                for root in roots:
                    f = root / f"{m}_sweep-lr{tag}" / "seed_42" / "metrics.json"
                    if f.is_file():
                        acc = dev_acc(json.loads(f.read_text(encoding="utf-8")))
                        if acc is None:
                            sys.exit(f"không thấy dev acc trong {f.name} của {m} lr={tag}; sửa DEV_KEYS")
                        res.setdefault(m, {})[lr] = acc
        sel, problems = {}, []
        for m in ALL_MODELS:
            got = res.get(m, {})
            grid = expected_lrs(m)
            if sorted(got) != grid:
                problems.append(f"{m}: có {len(got)}/{len(grid)} LR (thiếu {[l for l in grid if l not in got]}); cần zip của round 2 và round 5")
                continue
            best = max(got, key=lambda k: (got[k], -k))
            sel[m] = best
            edge = best in (grid[0], grid[-1])
            print(f"{m:14} " + "  ".join(f"{LR_TAG[l]}={got[l]:.4f}" for l in grid) + f"  -> {best:g}" + ("  [RÌA LƯỚI]" if edge else ""))
            if edge:
                problems.append(f"{m}: LR tốt nhất {best:g} ở rìa lưới -> mở thêm một bậc và ghi DEVIATIONS.md (PREREG)")
    for p in problems:
        print("CHÚ Ý:", p, file=sys.stderr)
    if any("có " in p and "/" in p for p in problems):
        return 1
    Path(a.out).write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("đã ghi", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
