#!/usr/bin/env python3
"""Kế hoạch chạy Mức B (docs/PLAN.md mục 3): sinh, kiểm tra và đọc scripts/run_plan.json.

Lệnh:
  generate [--out FILE]        sinh run_plan.json (xác định; cần pyyaml để đọc configs/estimates.yaml)
  check [--regen] [--tag T]    kiểm run_plan.json và kaggle-run.yml (sessions, STAGE); --regen: sinh lại và so băm;
                               --tag: tag release phải dạng r<N>[.<k>] với N = round của STAGE (smoke=r1, sweep=r2, p0a=r3, explore=r4, sweep2=r5, sweep3=r6, p0b=r7, xai=r8, multiseed=r9; dry-* để diễn tập)
  set-stage STAGE              sửa kaggle-run.yml (sessions và STAGE=) cho đúng stage; KHÔNG tự đoán stage kế tiếp
  summary                      bảng stage / session / GPU-giờ
  session-jobs --stage S --session N   TSV cho run_kaggle_food101.sh
  stage-ids --stage S          liệt kê id của stage (cho audit)

Mọi ước tính thời gian là "chưa đo" cho tới khi có job_times.csv từ smoke.
Không dùng hash() của Python (không xác định giữa các tiến trình): chỉ sha256 trên JSON chuẩn tắc.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT / "scripts" / "run_plan.json"
EST_PATH = ROOT / "configs" / "estimates.yaml"
MANIFEST_PATH = ROOT / "kaggle-run.yml"
LR_PATH = ROOT / "configs" / "selected_lr.json"

SEEDS = [42, 123, 2024]
SWEEP_SEED = 42
STAGE_ORDER = ["smoke", "sweep", "p0a", "explore", "sweep2", "sweep3", "p0b", "xai", "multiseed"]
# Round: smoke=r1 sweep=r2 p0a=r3 explore=r4 sweep2=r5 sweep3=r6 p0b=r7. sweep2/sweep3 = sweep mở rộng của r2 (DEVIATIONS D2/D4);
# multiseed = r9: thêm seed 123 và 2024 cho các cấu hình thăm dò vốn chỉ có seed 42 (DEVIATIONS D9).
# xai = r8: huấn luyện lại seed 42 (ViT-B/L, ConvNeXt-B) + checkpoint fp16 + XAI (DEVIATIONS D8).
# p0 tách thành p0a (LR đã chốt: ViT-B/L) và p0b (chờ sweep3). DEVIATIONS D5.
# Số round cố định theo stage (quy ước nhóm: tag release r<N>, zip <project>_r<N>_s<M>.zip)
STAGE_ROUND = {s: i + 1 for i, s in enumerate(STAGE_ORDER)}
SWEEP_BASE_LRS = {1e-5: "1e-5", 3e-5: "3e-5", 1e-4: "1e-4"}
SWEEP_EXT_LRS = {3e-4: "3e-4", 1e-3: "1e-3"}           # các LR của sweep2
SWEEP3_EXT_LRS = {1e-3: "1e-3", 3e-3: "3e-3"}          # các LR của sweep3
LR_TAG = {**SWEEP_BASE_LRS, **SWEEP_EXT_LRS, **SWEEP3_EXT_LRS}
# LR mở thêm (PREREG: best ở rìa lưới -> mở thêm một bậc), chọn sau kết quả round 2: 7 model có best=1e-4.
# ResNet-50 chỉ đạt 14% ở 1e-4 nên mở hai bậc ngay (tối đa 2 bậc). DINOv2/CLIP best ở trong lưới: không mở.
SWEEP2_EXT = {"vit_tiny": [3e-4], "vit_small": [3e-4], "vit_base": [3e-4], "vit_large": [3e-4],
              "convnext_base": [3e-4], "resnet50": [3e-4, 1e-3], "deit_base": [3e-4]}


# Sau sweep2: ViT-Ti/S, ConvNeXt-B, DeiT-B vẫn best=3e-4 (rìa), ResNet-50 best=1e-3 (rìa, còn tăng mạnh). Đây là lần mở rộng CUỐI:
# sau sweep3 model nào vẫn ở rìa thì chấp nhận và ghi Threats to validity (DEVIATIONS D4), không mở thêm.
SWEEP3_EXT = {"vit_tiny": [1e-3], "vit_small": [1e-3], "convnext_base": [1e-3], "deit_base": [1e-3], "resnet50": [3e-3]}
READY_MODELS = ["vit_base", "vit_large", "dinov2_base", "clip_base"]   # LR đã chốt trong lưới sau sweep + sweep2


def expected_lrs(model: str, upto: str = "sweep3") -> list:
    """Lưới LR mà model phải có đủ kết quả sweep trước khi chọn LR (gồm phần mở rộng)."""
    ext = SWEEP2_EXT.get(model, []) + (SWEEP3_EXT.get(model, []) if upto == "sweep3" else [])
    return sorted(list(SWEEP_BASE_LRS) + ext)

FT_SCALE = ["vit_tiny", "vit_small", "vit_base", "vit_large"]
FT_ARCH = ["convnext_base", "resnet50"]
EXPLORE_MODELS = ["deit_base", "dinov2_base", "clip_base"]
ALL_MODELS = FT_SCALE + FT_ARCH + EXPLORE_MODELS

STAGE_DESC = {
    "smoke": "20 bước thật/mô hình (--max-steps, dữ liệu thật) + eval đủ dev/test: đo throughput, bộ nhớ, kiểm đường ống. Không dùng cho kết quả.",
    "sweep": "LR sweep proxy chung: 20% train, 2 epoch, {1e-5,3e-5,1e-4}, seed 42, chọn theo dev acc.",
    "p0a": "Bảng chính, phần LR đã chốt: Full FT ViT-B/L x3 seed; LP ViT-B/L x3; LP-FT ViT-B x3, ViT-L x1.",
    "p0b": "Bảng chính, phần chờ sweep3: Full FT ViT-Ti/S, ConvNeXt-B, ResNet-50 x3 seed; thêm DeiT-B FT (thăm dò, 1 seed).",
    "explore": "Thăm dò (1 seed, không kiểm định): DINOv2/CLIP FT+LP, DeiT LP, CLIP zero-shot (quickgelu).",
    "sweep3": "Sweep mở rộng lần cuối (D4): 1e-3 cho ViT-Ti/S, ConvNeXt-B, DeiT-B; 3e-3 cho ResNet-50; proxy giống sweep.",
    "xai": "RQ3: huấn luyện lại Full FT seed 42 (ViT-B, ViT-L, ConvNeXt-B) với LR đã chốt, lưu checkpoint fp16, chạy XAI (rollout, Grad-CAM, IG; deletion/insertion; sanity) trên mẫu test cố định.",
    "multiseed": "Thêm seed 123 và 2024 cho các cấu hình fine-tuning chỉ có seed 42: DINOv2-B, CLIP-B, DeiT-B (đồ án chỉ báo cáo fine-tuning).",
    "sweep2": "Sweep mở rộng của round 2 (PREREG: LR tốt nhất ở rìa lưới): 3e-4 cho 7 model, thêm 1e-3 cho ResNet-50; proxy giống sweep (20% train, 2 epoch, seed 42).",
}
STAGE_DEPENDS = {"smoke": None, "sweep": "smoke", "p0a": "sweep2", "explore": "sweep2", "sweep2": "sweep", "sweep3": "sweep2", "p0b": "sweep3", "xai": "sweep3", "multiseed": "sweep3"}

PROTOCOL = {  # đưa vào config_hash: đổi giao thức => đổi băm
    "epochs": 5, "batch_effective": 128, "optimizer": "AdamW", "weight_decay": 0.05,
    "schedule": "cosine, warmup 10%", "amp": "fp16", "label_smoothing": 0.1,
    "dev_split": "10% train phân tầng, seed 1234",
}


# ---------------------------------------------------------------- định nghĩa job
SMOKE_STEPS = 20
SMOKE_FRAC = 0.05
SMOKE_LR = 3e-5   # chỉ để đường ống chạy (run_job đòi --lr cho ft/lpft); không dùng cho kết quả


XAI_MIN = {"vit_base": 15.0, "convnext_base": 15.0, "vit_large": 50.0}   # thời gian XAI mỗi job, CHƯA ĐO (ước từ số lần forward/backward fp32)


def _job(model, regime, seed, *, lr=None, epochs=None, train_frac=None, smoke=False, tag="", max_steps=None, xai=False):
    """lr: số, hoặc "selected" (đọc configs/selected_lr.json lúc chạy), hoặc None."""
    jid = f"{model}__{regime}__s{seed}"
    out = f"{model}_{regime}/seed_{seed}"
    if regime == "sweep":
        jid += f"__lr{tag}"
        out = f"{model}_sweep-lr{tag}/seed_{seed}"
    if xai:
        jid = f"{model}__{regime}-xai__s{seed}"
        out = f"{model}_{regime}-xai/seed_{seed}"
    if smoke:
        jid += "__smoke"
        out = f"smoke/{out}"
    job = {"id": jid, "model": model, "regime": regime, "seed": seed, "lr": lr,
           "epochs": epochs, "train_frac": train_frac, "max_steps": max_steps, "smoke": smoke, "out_rel": out}
    if xai:   # chỉ job xai mới có khoá này: không đổi config_hash/plan của các job cũ
        job["xai"] = True
    return job


def _smoke(model, regime):
    return _job(model, regime, 42, lr=SMOKE_LR if regime in ("ft", "lpft") else None, epochs=1,
                train_frac=SMOKE_FRAC, max_steps=SMOKE_STEPS, smoke=True)


def stage_jobs(stage: str) -> list[dict]:
    """Danh sách job theo thứ tự ưu tiên (seed-major: bị cắt giờ thì vẫn đủ seed đầu)."""
    jobs: list[dict] = []
    if stage == "smoke":
        for m in ALL_MODELS:
            jobs.append(_smoke(m, "ft"))
        jobs.append(_smoke("vit_base", "lp"))
        jobs.append(_smoke("vit_base", "lpft"))
        jobs.append(_smoke("clip_base", "zeroshot"))
    elif stage == "sweep":
        for lr, tag in SWEEP_BASE_LRS.items():
            for m in ALL_MODELS:
                jobs.append(_job(m, "sweep", SWEEP_SEED, lr=lr, epochs=2, train_frac=0.2, tag=tag))
    elif stage == "sweep2":
        for lr, tag in SWEEP_EXT_LRS.items():
            for m in ALL_MODELS:
                if lr in SWEEP2_EXT.get(m, []):
                    jobs.append(_job(m, "sweep", SWEEP_SEED, lr=lr, epochs=2, train_frac=0.2, tag=tag))
    elif stage == "sweep3":
        for lr, tag in SWEEP3_EXT_LRS.items():
            for m in ALL_MODELS:
                if lr in SWEEP3_EXT.get(m, []):
                    jobs.append(_job(m, "sweep", SWEEP_SEED, lr=lr, epochs=2, train_frac=0.2, tag=tag))
    elif stage == "p0a":
        for s in SEEDS:
            for m in ("vit_base", "vit_large"):
                jobs.append(_job(m, "ft", s, lr="selected"))
                jobs.append(_job(m, "lp", s))
            jobs.append(_job("vit_base", "lpft", s, lr="selected"))
            if s == 42:
                jobs.append(_job("vit_large", "lpft", s, lr="selected"))
    elif stage == "p0b":
        for s in SEEDS:
            for m in ("vit_tiny", "vit_small", "convnext_base", "resnet50"):
                jobs.append(_job(m, "ft", s, lr="selected"))
        jobs.append(_job("deit_base", "ft", 42, lr="selected"))
    elif stage == "explore":
        for m in ("dinov2_base", "clip_base"):
            jobs.append(_job(m, "ft", 42, lr="selected"))
            jobs.append(_job(m, "lp", 42))
        jobs.append(_job("deit_base", "lp", 42))
        jobs.append(_job("clip_base", "zeroshot", 42))
    elif stage == "multiseed":
        for s_ in SEEDS[1:]:
            for m in ("dinov2_base", "clip_base", "deit_base"):
                jobs.append(_job(m, "ft", s_, lr="selected"))
    elif stage == "xai":
        for m in ("vit_base", "convnext_base", "vit_large"):
            jobs.append(_job(m, "ft", 42, lr="selected", xai=True))
    else:
        raise ValueError(stage)
    return jobs


# ---------------------------------------------------------------- ước tính
def estimate_minutes(job: dict, est: dict) -> float:
    """Phút 1 GPU cho job (chưa đo). Chưa nhân safety_factor."""
    p = est["protocol"]
    thr = float(est["throughput_train_img_s"][job["model"]])
    ratio = float(est["eval_to_train_ratio"])
    inf = thr * ratio
    n_tr, n_dev, n_te = p["train_images"], p["dev_images"], p["test_images"]
    over = float(est["job_overhead_min"])
    reg = job["regime"]
    if job["smoke"]:
        steps, sfrac = int(est["smoke_steps"]), float(est["smoke_train_frac"])
        evals = (n_dev + n_te) / inf
        train = steps * p["batch_effective"] / thr if reg in ("ft", "lpft") else 0.0
        extract = (n_tr * sfrac) / inf if reg in ("lp", "lpft") else 0.0
        return round((train + evals + extract) / 60 + over + float(est["smoke_extra_min"]), 1)
    epochs = job["epochs"] or p["epochs"]
    frac = job["train_frac"] or 1.0

    def lp_min():
        return (n_tr + n_dev + n_te) / inf / 60 + float(est["lp_head_min"])

    def ft_min(with_test=True):
        train = n_tr * frac * epochs / thr
        ev = (n_dev * epochs + (n_te if with_test else 0)) / inf
        return (train + ev) / 60

    if reg == "ft":
        m = ft_min()
    elif reg == "sweep":
        m = ft_min(with_test=False)
    elif reg == "lp":
        m = lp_min()
    elif reg == "lpft":
        m = lp_min() + ft_min()
    elif reg == "zeroshot":
        m = (n_te + n_dev) / inf / 60 + 1.0
    else:
        raise ValueError(reg)
    if job.get("xai"):
        m += XAI_MIN[job["model"]]
    return round(m + over, 1)


def config_hash(job: dict) -> str:
    core = {k: job[k] for k in ("model", "regime", "seed", "lr", "epochs", "train_frac", "max_steps", "smoke")}
    if job.get("xai"):
        core["xai"] = True
    core["protocol"] = PROTOCOL
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- xếp session
def _group(job: dict) -> int:
    """ViT-L huấn luyện trên toàn bộ train đặt session riêng (nhóm 1)."""
    big = job["model"] == "vit_large" and not job["smoke"] and (job["train_frac"] or 1.0) >= 1.0
    return 1 if big else 0


def _pack(items: list[dict], n_lanes: int, cap_min: float) -> list[list[list[dict]]]:
    """LPT cân bằng. Trả về [session][lane][job]. Số session nhỏ nhất mà mọi lane <= cap."""
    for j in items:
        if j["_safe_min"] > cap_min:
            raise SystemExit(f"job {j['id']} ({j['_safe_min']:.0f} phút) vượt sức chứa một lane ({cap_min:.0f} phút)")
    total = sum(j["_safe_min"] for j in items)
    n = max(1, math.ceil(total / (cap_min * n_lanes)))
    order = sorted(items, key=lambda j: (-j["_safe_min"], j["id"]))
    while True:
        loads = [[0.0] * n_lanes for _ in range(n)]
        lanes: list[list[list[dict]]] = [[[] for _ in range(n_lanes)] for _ in range(n)]
        ok = True
        for j in order:
            si, li = min(((s, l) for s in range(n) for l in range(n_lanes)), key=lambda t: (loads[t[0]][t[1]], t))
            if loads[si][li] + j["_safe_min"] > cap_min:
                ok = False
                break
            loads[si][li] += j["_safe_min"]
            lanes[si][li].append(j)
        if ok:
            return lanes
        n += 1


def build_stage(stage: str, est: dict) -> dict:
    n_lanes = int(est["n_lanes"])
    safety = float(est["safety_factor"])
    cap = float(est["lane_cap_hours"]) * 60 - float(est["session_overhead_min"])
    jobs = stage_jobs(stage)
    for prio, j in enumerate(jobs):
        j["_prio"] = prio
        j["est_min"] = estimate_minutes(j, est)
        j["_safe_min"] = round(j["est_min"] * safety, 1)
        j["config_hash"] = config_hash(j)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), f"id trùng trong stage {stage}"
    sessions = []
    for grp in (0, 1):
        items = [j for j in jobs if _group(j) == grp]
        if not items:
            continue
        for lanes in _pack(items, n_lanes, cap):
            lanes = [sorted(l, key=lambda j: j["_prio"]) for l in lanes]
            sessions.append({
                "session": len(sessions) + 1,
                "group": "vit_large" if grp else "main",
                "lanes": [[j["id"] for j in l] for l in lanes],
                "est_lane_min": [round(sum(j["est_min"] for j in l), 1) for l in lanes],
                "est_lane_min_safe": [round(sum(j["_safe_min"] for j in l), 1) for l in lanes],
            })
    gpu_min = sum(j["est_min"] for j in jobs)
    return {
        "round": STAGE_ROUND[stage],
        "description": STAGE_DESC[stage],
        "depends_on": STAGE_DEPENDS[stage],
        "requires_selected_lr": any(j["lr"] == "selected" for j in jobs),
        "n_jobs": len(jobs),
        "n_sessions": len(sessions),
        "est_gpu_hours": round(gpu_min / 60, 2),
        "sessions": sessions,
        "jobs": [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs],
    }


def load_estimates() -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("cần pyyaml để đọc configs/estimates.yaml (pip install pyyaml)")
    return yaml.safe_load(EST_PATH.read_text(encoding="utf-8"))


def plan_hash(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_plan(est: dict) -> dict:
    stages = {s: build_stage(s, est) for s in STAGE_ORDER}
    plan = {
        "schema": 1,
        "project": "food101-vit",
        "note": "ĐÓNG BĂNG. Sinh bởi scripts/plan.py từ configs/estimates.yaml. Ước tính thời gian: chưa đo.",
        "estimates_status": est.get("status", "chưa đo"),
        "seeds": SEEDS,
        "protocol": PROTOCOL,
        "stage_order": STAGE_ORDER,
        "estimates": {k: est[k] for k in ("throughput_train_img_s", "eval_to_train_ratio", "job_overhead_min",
                                          "safety_factor", "lane_cap_hours", "session_overhead_min", "n_lanes")},
        "stages": stages,
    }
    plan["plan_hash"] = plan_hash(plan)
    return plan


def dump(plan: dict) -> str:
    return json.dumps(plan, indent=1, ensure_ascii=False) + "\n"


def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- manifest
def read_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Đọc sessions / STAGE / budget_hours / notebook bằng regex, không cần pyyaml (chạy được trên runner)."""
    txt = path.read_text(encoding="utf-8")
    out = {}
    for key in ("sessions", "budget_hours", "notebook"):
        m = re.search(rf"^{key}:\s*([^\s#]+)", txt, re.M)
        out[key] = m.group(1) if m else None
    m = re.search(r"\bSTAGE=([A-Za-z0-9_]+)", txt)
    out["stage"] = m.group(1) if m else None
    return out


def cmd_check(args) -> int:
    errs: list[str] = []
    plan = load_plan()
    if plan_hash(plan) != plan.get("plan_hash"):
        errs.append("run_plan.json bị sửa tay (plan_hash không khớp); sinh lại bằng plan.py generate")
    if args.regen:
        fresh = build_plan(load_estimates())
        if fresh["plan_hash"] != plan["plan_hash"]:
            errs.append("run_plan.json khác kết quả sinh lại từ plan.py/estimates.yaml (chưa đóng băng lại?)")
    man = read_manifest()
    st = man["stage"]
    if st not in plan["stages"]:
        errs.append(f"kaggle-run.yml: STAGE={st!r} không có trong run_plan.json {list(plan['stages'])}")
    else:
        n = plan["stages"][st]["n_sessions"]
        if str(n) != str(man["sessions"]):
            errs.append(f"kaggle-run.yml sessions={man['sessions']} != n_sessions={n} của stage {st}")
    if not re.fullmatch(r"[a-z0-9-]{5,50}", man["notebook"] or ""):
        errs.append(f"slug notebook không hợp lệ: {man['notebook']!r}")
    try:
        if float(man["budget_hours"]) > 11.5:
            errs.append("budget_hours > 11.5")
    except (TypeError, ValueError):
        errs.append(f"budget_hours không đọc được: {man['budget_hours']!r}")
    tag = getattr(args, "tag", "") or ""
    rnd = STAGE_ROUND.get(st)
    if tag and not tag.startswith("dry-") and not re.fullmatch(rf"r{rnd}(\.\d+)?", tag):
        errs.append(f"tag {tag!r} không khớp round của stage {st!r} (round {rnd}): dùng r{rnd} hoặc r{rnd}.<k> khi gửi lại; dry-* để diễn tập)")
    cap = plan["estimates"]["lane_cap_hours"]
    try:
        if float(man["budget_hours"]) != float(cap):
            errs.append(f"budget_hours={man['budget_hours']} khác lane_cap_hours={cap} dùng khi xếp session")
    except (TypeError, ValueError):
        pass
    for sname, s in plan["stages"].items():
        seen = [i for sess in s["sessions"] for lane in sess["lanes"] for i in lane]
        want = [j["id"] for j in s["jobs"]]
        if sorted(seen) != sorted(want):
            errs.append(f"stage {sname}: job trong session không khớp danh sách job")
        if [x["session"] for x in s["sessions"]] != list(range(1, s["n_sessions"] + 1)):
            errs.append(f"stage {sname}: số thứ tự session không liên tục")
        for sess in s["sessions"]:
            if not any(sess["lanes"]):
                errs.append(f"stage {sname} session {sess['session']} rỗng")
    if st in plan["stages"] and plan["stages"][st]["requires_selected_lr"]:   # chặn từ GitHub, không đợi session Kaggle lỗi
        if not LR_PATH.exists():
            errs.append(f"stage {st} cần configs/selected_lr.json (chạy scripts/pick_lr.py sau sweep rồi commit)")
        else:
            sel = json.loads(LR_PATH.read_text(encoding="utf-8"))
            need = sorted({j["model"] for j in plan["stages"][st]["jobs"] if j["lr"] == "selected"})
            miss = [m for m in need if m not in sel]
            if miss:
                errs.append(f"stage {st}: configs/selected_lr.json thiếu LR của {miss}")
    for e in errs:
        print("LỖI:", e, file=sys.stderr)
    if not errs:
        print(f"OK: stage={st} sessions={man['sessions']} khớp run_plan.json (plan_hash {plan['plan_hash']})")
    return 1 if errs else 0


def cmd_set_stage(args) -> int:
    plan = load_plan()
    st = args.stage
    if st not in plan["stages"]:
        sys.exit(f"stage không có: {st}; chọn một trong {list(plan['stages'])}")
    n = plan["stages"][st]["n_sessions"]
    txt = MANIFEST_PATH.read_text(encoding="utf-8")
    txt, k1 = re.subn(r"^(sessions:\s*)\d+", rf"\g<1>{n}", txt, flags=re.M)
    txt, k2 = re.subn(r"\bSTAGE=[A-Za-z0-9_]+", f"STAGE={st}", txt)
    if k1 != 1 or k2 != 1:
        sys.exit("kaggle-run.yml không đúng khuôn (cần đúng một dòng 'sessions:' và một 'STAGE=')")
    MANIFEST_PATH.write_text(txt, encoding="utf-8")
    print(f"kaggle-run.yml: STAGE={st}, sessions={n}. Chưa commit, chưa tạo release.")
    return cmd_check(argparse.Namespace(regen=False))


def cmd_summary(args) -> int:
    plan = load_plan()
    tot = 0.0
    print(f"plan_hash {plan['plan_hash']}  (ước tính: {plan['estimates_status']})")
    print(f"{'round':5} {'stage':8} {'job':>4} {'session':>7} {'GPU-giờ':>8}  wall/session (lane, đã nhân safety)")
    for name in plan["stage_order"]:
        s = plan["stages"][name]
        tot += s["est_gpu_hours"]
        walls = ", ".join(f"s{x['session']}[{x['group']}] {max(x['est_lane_min_safe'])/60:.1f}h" for x in s["sessions"])
        print(f"r{s['round']:<4} {name:8} {s['n_jobs']:>4} {s['n_sessions']:>7} {s['est_gpu_hours']:>8.2f}  {walls}")
    print(f"TỔNG GPU-giờ (ước tính chưa đo, không gồm smoke {plan['stages']['smoke']['est_gpu_hours']:.2f}): "
          f"{tot - plan['stages']['smoke']['est_gpu_hours']:.2f}")
    return 0


def cmd_session_jobs(args) -> int:
    plan = load_plan()
    if args.stage not in plan["stages"]:
        sys.exit(f"stage không có: {args.stage}")
    st = plan["stages"][args.stage]
    if not 1 <= args.session <= st["n_sessions"]:
        print(f"session {args.session} ngoài 1..{st['n_sessions']} của stage {args.stage} (sessions trong manifest lệch plan)",
              file=sys.stderr)
        return 2
    sess = st["sessions"][args.session - 1]
    byid = {j["id"]: j for j in st["jobs"]}
    sel = {}
    if st["requires_selected_lr"]:
        if not LR_PATH.exists():
            print(f"thiếu {LR_PATH.relative_to(ROOT)}: chạy stage sweep rồi scripts/pick_lr.py trước", file=sys.stderr)
            return 4
        sel = json.loads(LR_PATH.read_text(encoding="utf-8"))
    rows = []
    for li, lane in enumerate(sess["lanes"]):
        for jid in lane:
            j = byid[jid]
            a = ["--model", j["model"], "--regime", j["regime"], "--seed", str(j["seed"])]
            lr = j["lr"]
            if lr == "selected":
                if j["model"] not in sel:
                    print(f"thiếu LR đã chọn cho {j['model']} trong configs/selected_lr.json", file=sys.stderr)
                    return 4
                lr = sel[j["model"]]
            if lr is not None:
                a += ["--lr", repr(float(lr))]
            if j["epochs"]:
                a += ["--epochs", str(j["epochs"])]
            if j["train_frac"]:
                a += ["--train-frac", str(j["train_frac"])]
            if j.get("max_steps"):
                a += ["--max-steps", str(j["max_steps"])]
            if j.get("xai"):
                a += ["--xai"]
            rows.append((jid, li, j["est_min"], j["out_rel"], " ".join(a)))
    for r in rows:
        print("\t".join(str(x) for x in r))
    return 0


def cmd_round(args) -> int:
    print(STAGE_ROUND[args.stage])
    return 0


def cmd_stage_ids(args) -> int:
    for j in load_plan()["stages"][args.stage]["jobs"]:
        print(j["id"])
    return 0


def cmd_generate(args) -> int:
    plan = build_plan(load_estimates())
    out = Path(args.out) if args.out else PLAN_PATH
    out.write_text(dump(plan), encoding="utf-8")
    print(f"đã ghi {out} (plan_hash {plan['plan_hash']})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out"); g.set_defaults(fn=cmd_generate)
    c = sub.add_parser("check"); c.add_argument("--regen", action="store_true"); c.add_argument("--tag", default=""); c.set_defaults(fn=cmd_check)
    s = sub.add_parser("set-stage"); s.add_argument("stage"); s.set_defaults(fn=cmd_set_stage)
    sub.add_parser("summary").set_defaults(fn=cmd_summary)
    sj = sub.add_parser("session-jobs"); sj.add_argument("--stage", required=True)
    sj.add_argument("--session", type=int, required=True); sj.set_defaults(fn=cmd_session_jobs)
    rd = sub.add_parser("round"); rd.add_argument("--stage", required=True); rd.set_defaults(fn=cmd_round)
    si = sub.add_parser("stage-ids"); si.add_argument("--stage", required=True); si.set_defaults(fn=cmd_stage_ids)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
