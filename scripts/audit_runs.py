#!/usr/bin/env python3
"""Kiểm kết quả Food-101 trước khi đưa vào bảng (lược đồ Food-101; viết lại, không dùng bản ACSA).

Kiểm, theo scripts/run_plan.json: (1) đủ job/seed, báo THIẾU và THỪA; (2) đủ file; (3) test_predictions.jsonl đủ
25,250 dòng, nhãn và dự đoán trong 0..100, không trùng chỉ số; (4) logits hữu hạn, đúng hình (N,101); (5) metrics.json không NaN;
(6) acc tính lại từ test_logits khớp metrics.json và khớp cột dự đoán; (7) cùng một thứ tự nhãn test ở mọi run.

  python3 scripts/audit_runs.py --runs runs_inbox/food101-ci/p0/s1-v1/food101_r3_s1.zip runs_inbox/.../food101_r3_s2.zip --stage p0a
  python3 scripts/audit_runs.py --runs DIR_CHUA_outputs --stage p0a p0b explore

Lược đồ khớp food101.run_job/evaluate.py (jsonl: id, gold, pred, top5_*; metrics: {"dev":{"top1"...},"test":{"top1"...}});
vẫn chấp nhận vài tên khoá khác (xem *_KEYS). Thoát 1 nếu có lỗi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABEL_KEYS = ("gold", "label", "y", "target", "gt", "true_label")
PRED_KEYS = ("pred", "prediction", "pred_label", "y_pred", "top1")
IDX_KEYS = ("idx", "index", "id", "image_id", "i")
ACC_KEYS = ("test_acc", "test_top1", "top1", "acc", "accuracy", "test_accuracy", "top1_acc")

REQ = {  # regime -> file bắt buộc
    "ft": ["metrics.json", "history.json", "run_config.yaml", "test_predictions.jsonl", "test_logits.npy", "dev_logits.npy"],
    "lp": ["metrics.json", "history.json", "run_config.yaml", "test_predictions.jsonl", "test_logits.npy", "dev_logits.npy"],
    "lpft": ["metrics.json", "history.json", "run_config.yaml", "test_predictions.jsonl", "test_logits.npy", "dev_logits.npy"],
    "zeroshot": ["metrics.json", "run_config.yaml", "test_predictions.jsonl", "test_logits.npy"],
    "sweep": ["metrics.json", "history.json"],
    "bench": ["bench.json", "run_config.yaml"],   # D10: không có logits/predictions
}


def extract(zpath: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            t = (dest / n).resolve()
            if not str(t).startswith(str(dest.resolve())):
                raise SystemExit(f"zip có đường dẫn nguy hiểm: {n}")
        z.extractall(dest)
    return dest


def find_outputs(p: Path) -> Path:
    return p / "outputs" if (p / "outputs").is_dir() else p


def flatten(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f"{prefix}{k}.")
    else:
        yield prefix[:-1], d


def find_acc(metrics: dict):
    """Trả về (đường dẫn, giá trị) của acc test trong metrics.json; ưu tiên khoá có 'test'."""
    cands = [(k, v) for k, v in flatten(metrics) if k.split(".")[-1] in ACC_KEYS and isinstance(v, (int, float))]
    cands.sort(key=lambda kv: (0 if "test" in kv[0] else 1, len(kv[0])))
    return cands[0] if cands else (None, None)


def _list_nan(v) -> bool:
    if isinstance(v, list):
        return any(_list_nan(x) for x in v)
    if isinstance(v, dict):
        return any(_list_nan(x) for x in v.values())
    return isinstance(v, float) and not math.isfinite(v)


def pick(row: dict, keys):
    for k in keys:
        if k in row:
            return k
    return None


def read_predictions(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"dòng {ln} không phải JSON: {e}")
    return rows


def audit_job(job: dict, d: Path, n_test: int, n_dev: int, n_cls: int, tol: float, probs: list, warns: list):
    jid, reg = job["id"], job["regime"]
    P = lambda m: probs.append(f"{jid}: {m}")
    W = lambda m: warns.append(f"{jid}: {m}")
    info = {"acc_logits": None}
    for fn in REQ[reg]:
        if not (d / fn).is_file():
            P(f"thiếu file {fn}")
    if reg == "bench":   # chỉ kiểm bench.json: hữu hạn, params > 0, đủ khoá
        if (d / "bench.json").is_file():
            try:
                b = json.loads((d / "bench.json").read_text(encoding="utf-8"))
                if _list_nan(b):
                    P("bench.json có NaN/Inf")
                for k in ("params", "gmacs", "latency_b1_fp32", "device"):
                    if k not in b:
                        P(f"bench.json thiếu khoá {k}")
                if b.get("device") == "cuda" and "latency_b1_fp16_autocast" not in b:
                    P("bench.json (cuda) thiếu latency fp16")
                if b.get("device") != "cuda":
                    W("bench chạy trên CPU, không phải GPU T4")
            except json.JSONDecodeError as e:
                P(f"bench.json hỏng: {e}")
        return info
    metrics = None
    if (d / "metrics.json").is_file():
        try:
            metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            if _list_nan(metrics):
                P("metrics.json có NaN/Inf")
        except json.JSONDecodeError as e:
            P(f"metrics.json hỏng: {e}")
    if (d / "history.json").is_file():
        try:
            if _list_nan(json.loads((d / "history.json").read_text(encoding="utf-8"))):
                P("history.json có NaN/Inf (loss phân kỳ?)")
        except json.JSONDecodeError as e:
            P(f"history.json hỏng: {e}")
    if reg == "sweep":
        return info
    logits = None
    if (d / "test_logits.npy").is_file():
        try:
            logits = np.load(d / "test_logits.npy")
            if logits.shape != (n_test, n_cls):
                P(f"test_logits.npy hình {logits.shape}, cần {(n_test, n_cls)}")
                logits = None
            elif not np.isfinite(logits).all():
                P("test_logits.npy có NaN/Inf")
        except Exception as e:  # noqa: BLE001
            P(f"test_logits.npy không đọc được: {e}")
    if (d / "dev_logits.npy").is_file():
        try:
            dl = np.load(d / "dev_logits.npy")
            if dl.shape != (n_dev, n_cls):
                P(f"dev_logits.npy hình {dl.shape}, cần {(n_dev, n_cls)}")
            elif not np.isfinite(dl).all():
                P("dev_logits.npy có NaN/Inf")
        except Exception as e:  # noqa: BLE001
            P(f"dev_logits.npy không đọc được: {e}")
    labels = None
    if (d / "test_predictions.jsonl").is_file():
        try:
            rows = read_predictions(d / "test_predictions.jsonl")
        except ValueError as e:
            P(str(e)); rows = []
        if len(rows) != n_test:
            P(f"test_predictions.jsonl có {len(rows)} dòng, cần {n_test}")
        if rows:
            lk, pk, ik = pick(rows[0], LABEL_KEYS), pick(rows[0], PRED_KEYS), pick(rows[0], IDX_KEYS)
            if lk is None or pk is None:
                P(f"không nhận ra khoá nhãn/dự đoán trong jsonl (khoá có: {sorted(rows[0])}); sửa LABEL_KEYS/PRED_KEYS")
            else:
                try:
                    lab = np.array([r[lk] for r in rows]); prd = np.array([r[pk] for r in rows])
                    if np.issubdtype(lab.dtype, np.floating) and not np.isfinite(lab).all() or \
                       np.issubdtype(prd.dtype, np.floating) and not np.isfinite(prd).all():
                        P("nhãn/dự đoán có NaN")
                    elif ((lab < 0) | (lab > n_cls - 1)).any() or ((prd < 0) | (prd > n_cls - 1)).any():
                        P(f"nhãn hoặc dự đoán ngoài 0..{n_cls - 1}")
                    else:
                        labels = lab.astype(int)
                        if ik is not None:
                            idx = [r[ik] for r in rows]
                            if len(set(idx)) != len(idx):
                                P("chỉ số mẫu trùng trong predictions")
                        if logits is not None and len(labels) == n_test:
                            am = logits.argmax(1)
                            acc = float((am == labels).mean())
                            info["acc_logits"] = acc
                            mism = float((am != prd.astype(int)).mean())
                            if mism > 0:
                                P(f"{mism:.4%} dự đoán trong jsonl khác argmax(test_logits)")
                            if metrics is not None:
                                path, val = find_acc(metrics)
                                if val is None:
                                    W("không thấy acc trong metrics.json (kiểm ACC_KEYS)")
                                else:
                                    v = val / 100 if val > 1.0 else val
                                    if abs(v - acc) > tol:
                                        P(f"acc metrics.json[{path}]={val} != acc tính lại từ logits={acc:.6f}")
                except (KeyError, ValueError, TypeError) as e:
                    P(f"jsonl không đều khoá/kiểu: {e}")
    info["labels_hash"] = hashlib.sha256(labels.tobytes()).hexdigest()[:12] if labels is not None else None
    return info


def audit(plan: dict, roots: list[Path], stages: list[str], n_test=25250, n_dev=7575, n_cls=101, tol=1e-4) -> dict:
    probs: list[str] = []
    warns: list[str] = []
    all_dirs: dict[str, Path] = {}  # out_rel -> thư mục
    for r in roots:
        o = find_outputs(r)
        for m in sorted(list(o.rglob("metrics.json")) + list(o.rglob("bench.json"))):
            rel = m.parent.relative_to(o).as_posix()
            if rel in all_dirs:
                warns.append(f"{rel}: có ở nhiều nguồn, dùng nguồn đầu")
            else:
                all_dirs[rel] = m.parent
    known = {j["out_rel"] for s in plan["stages"].values() for j in s["jobs"]}
    expected = [j for s in stages for j in plan["stages"][s]["jobs"]]
    exp_rel = {j["out_rel"] for j in expected}
    missing = [j["id"] for j in expected if j["out_rel"] not in all_dirs]
    extra = sorted(r for r in all_dirs if r not in known)
    other_stage = sorted(r for r in all_dirs if r in known and r not in exp_rel)
    for m in missing:
        probs.append(f"THIẾU job {m}")
    for e in extra:
        probs.append(f"THỪA thư mục không có trong run_plan: {e}")
    hashes = {}
    accs = {}
    for j in expected:
        d = all_dirs.get(j["out_rel"])
        if d is None:
            continue
        info = audit_job(j, d, n_test, n_dev, n_cls, tol, probs, warns)
        accs[j["id"]] = info["acc_logits"]
        if info.get("labels_hash"):
            hashes[j["id"]] = info["labels_hash"]
    if len(set(hashes.values())) > 1:
        probs.append(f"thứ tự/giá trị nhãn test KHÁC nhau giữa các run: {sorted(set(hashes.values()))}")
    # đủ seed theo (model, regime)
    by = {}
    for j in expected:
        by.setdefault((j["model"], j["regime"]), []).append(j)
    for (m, r), js in sorted(by.items()):
        got = [j["seed"] for j in js if j["out_rel"] in all_dirs]
        if len(got) != len(js):
            probs.append(f"{m}/{r}: có seed {sorted(got)}, plan cần {sorted(j['seed'] for j in js)}")
    return {"ok": not probs, "problems": probs, "warnings": warns, "n_expected": len(expected),
            "n_found": len(expected) - len(missing), "other_stage_dirs": other_stage, "acc_from_logits": accs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="thư mục (chứa outputs/) hoặc file zip, có thể nhiều")
    ap.add_argument("--stage", nargs="+", default=["p0a", "p0b", "explore"])
    ap.add_argument("--plan", default=str(ROOT / "scripts" / "run_plan.json"))
    ap.add_argument("--n-test", type=int, default=25250)
    ap.add_argument("--n-dev", type=int, default=7575)
    ap.add_argument("--n-classes", type=int, default=101)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--json", help="ghi báo cáo JSON")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    bad = [s for s in a.stage if s not in plan["stages"]]
    if bad:
        sys.exit(f"stage không có trong plan: {bad}")
    with tempfile.TemporaryDirectory(prefix="audit_") as tmp:
        roots = []
        for i, r in enumerate(a.runs):
            p = Path(r)
            if not p.exists():
                sys.exit(f"không có: {r}")
            roots.append(extract(p, Path(tmp) / f"z{i}") if p.suffix == ".zip" else p)
        rep = audit(plan, roots, a.stage, a.n_test, a.n_dev, a.n_classes, a.tol)
    print(f"stage {a.stage}: tìm thấy {rep['n_found']}/{rep['n_expected']} job")
    for w in rep["warnings"]:
        print("CẢNH BÁO:", w)
    for p in rep["problems"]:
        print("LỖI:", p)
    print("KẾT QUẢ:", "ĐẠT" if rep["ok"] else f"KHÔNG ĐẠT ({len(rep['problems'])} lỗi)")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
