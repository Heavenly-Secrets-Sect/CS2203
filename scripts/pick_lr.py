#!/usr/bin/env python3
"""Chọn LR theo DEV acc từ kết quả stage sweep (PREREG: không dùng test). Ghi configs/selected_lr.json.

  python3 scripts/pick_lr.py food101_r2_s1.zip food101_r5_s1.zip [...]   # round 2 + round 5 (sweep mở rộng) [--out configs/selected_lr.json] [--models ready|m1,m2] [--upto sweep2|sweep3]
LR tốt nhất ở rìa lưới => in cảnh báo: PREREG yêu cầu mở thêm một bậc cho model đó và ghi DEVIATIONS.md (cần plan mới).
Khoá dev acc trong metrics.json là giả định (DEV_KEYS); sửa nếu food101.run_job dùng tên khác.
"""
import argparse, json, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import ALL_MODELS, LR_TAG, LR_PATH, expected_lrs, READY_MODELS, NEW_MODELS  # noqa: E402

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
    ap.add_argument("--models", help="chỉ chọn cho các model này (vd. ready = vit_base,vit_large,dinov2_base,clip_base); mặc định: tất cả")
    ap.add_argument("--upto", default="sweep3", choices=["sweep2", "sweep3"], help="lưới kỳ vọng đến stage nào")
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
        # model mới (D10) chưa có kết quả thì bỏ qua khi không chỉ định --models
        want = [m for m in ALL_MODELS if m in res or m not in NEW_MODELS] if not a.models else (READY_MODELS if a.models == "ready" else a.models.split(","))
        for m in want:
            got = res.get(m, {})
            grid = expected_lrs(m, a.upto)
            if sorted(got) != grid:
                problems.append(f"{m}: có {len(got)}/{len(grid)} LR (thiếu {[l for l in grid if l not in got]}); cần đủ zip (model cũ: round 2, 5, 6; model mới Swin-B/MobileNetV3: round 10)")
                continue
            best = max(got, key=lambda k: (got[k], -k))
            sel[m] = best
            edge = best in (grid[0], grid[-1])
            print(f"{m:14} " + "  ".join(f"{LR_TAG[l]}={got[l]:.4f}" for l in grid) + f"  -> {best:g}" + ("  [RÌA LƯỚI]" if edge else ""))
            if edge and m in NEW_MODELS:   # D10: lưới rộng 6 bậc một lần, không mở thêm
                problems.append(f"{m}: LR tốt nhất {best:g} ở rìa lưới 6 bậc -> CHẤP NHẬN, không mở thêm; ghi vào hạn chế (DEVIATIONS D10)")
            elif edge:
                problems.append(f"{m}: LR tốt nhất {best:g} ở rìa lưới -> mở thêm một bậc và ghi DEVIATIONS.md (PREREG)")
    for p in problems:
        print("CHÚ Ý:", p, file=sys.stderr)
    if any("có " in p and "/" in p for p in problems):
        return 1
    if a.models and Path(a.out).is_file():   # chọn từng phần: giữ các model đã có
        old = json.loads(Path(a.out).read_text(encoding="utf-8"))
        old.update(sel); sel = old
    Path(a.out).write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("đã ghi", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
