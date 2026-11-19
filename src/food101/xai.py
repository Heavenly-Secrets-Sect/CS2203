"""XAI cho RQ3 (docs/PLAN.md mục 8): Attention Rollout (chỉ ViT, class-agnostic), Grad-CAM, Integrated Gradients (theo lớp).

Mọi bản đồ quy về lưới GRID x GRID (14x14 = lưới patch của ViT/16 ở 224) để so chung. Giải thích cho lớp DỰ ĐOÁN.
Đo deletion / insertion AUC (xác suất lớp dự đoán khi xoá / thêm dần patch theo thứ tự bản đồ, nền = ảnh trung bình, tức 0 sau chuẩn hoá)
trên mẫu cố định (2 ảnh/lớp test, seed cố định), và sanity check kiểu Adebayo: random hoá head + khối cuối rồi lấy Spearman với bản đồ gốc.
Giải thích là giả thuyết về hành vi model, không phải nhân quả.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

GRID = 14
SAMPLE_SEED = 20240601
N_STEPS = 15          # số mốc deletion/insertion (gồm 0% và 100%)
IG_STEPS = 16
IG_CHUNK = 8
N_DEMO = 24           # số ảnh lưu bản đồ để vẽ hình
METHODS = ("rollout", "gradcam", "ig")


def fixed_sample(labels: np.ndarray, per_class: int = 2, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Chỉ số test cố định, phân tầng theo lớp, sắp xếp tăng dần."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        idx.extend(rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist())
    return np.array(sorted(idx), dtype=np.int64)


def is_vit(model) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed")


def _to_grid(m):
    """(H, W) tensor -> (GRID, GRID) bằng average pooling thích nghi."""
    import torch.nn.functional as F
    return F.adaptive_avg_pool2d(m[None, None].float(), GRID)[0, 0]


# ------------------------------------------------------------------ Attention Rollout
def attention_rollout(model, x):
    """x: (1,3,H,W). Trả (GRID,GRID) hoặc None nếu không phải ViT. Abnar & Zuidema: A' = 0.5*A + 0.5*I, chuẩn hoá hàng, nhân dồn."""
    import torch
    if not is_vit(model):
        return None
    atts, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False  # để attn_drop thấy ma trận attention
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: atts.append(o.detach())))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    r = None
    for a in atts:
        a = a.mean(1)[0]                      # (T,T) trung bình các head
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    npre = getattr(model, "num_prefix_tokens", 1)
    cls_row = r[0, npre:]
    g = int(round(math.sqrt(cls_row.numel())))
    return _to_grid(cls_row.reshape(g, g))


# ------------------------------------------------------------------ Grad-CAM
def _gradcam_target(model):
    return model.blocks[-1].norm1 if is_vit(model) else model.stages[-1]


def gradcam(model, x, cls):
    import torch
    store = {}
    layer = _gradcam_target(model)

    def fwd(m, i, o):
        o.retain_grad()
        store["a"] = o
    h = layer.register_forward_hook(fwd)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out[0, cls].backward()
    finally:
        h.remove()
    a, g = store["a"], store["a"].grad
    if is_vit(model):
        npre = getattr(model, "num_prefix_tokens", 1)
        w = g[0, npre:].mean(0)                         # (D,)
        cam = torch.relu((a[0, npre:] * w).sum(-1))     # (T-npre,)
        s = int(round(math.sqrt(cam.numel())))
        cam = cam.reshape(s, s)
    else:
        w = g[0].mean((1, 2))                           # (C,)
        cam = torch.relu((a[0] * w[:, None, None]).sum(0))
    model.zero_grad(set_to_none=True)
    return _to_grid(cam.detach())


# ------------------------------------------------------------------ Integrated Gradients
def integrated_gradients(model, x, cls, steps: int = IG_STEPS):
    """Nền = 0 (ảnh trung bình sau chuẩn hoá). Gộp |attr| theo kênh."""
    import torch
    base = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)[:, None, None, None]
    grads = []
    for a in alphas.split(IG_CHUNK):                     # chia nhỏ để ViT-L fp32 vừa T4
        xs = (base + a * (x - base)).requires_grad_(True)
        model(xs)[:, cls].sum().backward()
        grads.append(xs.grad.detach())
    avg = torch.cat(grads).mean(0, keepdim=True)
    attr = ((x - base) * avg).abs().sum(1)[0]
    model.zero_grad(set_to_none=True)
    return _to_grid(attr.detach())


def compute_maps(model, x, cls):
    maps = {"gradcam": gradcam(model, x, cls), "ig": integrated_gradients(model, x, cls)}
    r = attention_rollout(model, x)
    if r is not None:
        maps["rollout"] = r
    return maps


# ------------------------------------------------------------------ deletion / insertion
def _auc(ys):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ys, np.linspace(0.0, 1.0, len(ys))))


def deletion_insertion(model, x, cls, grid_map, n_steps: int = N_STEPS):
    """Trả (deletion_auc, insertion_auc) của xác suất lớp `cls`. Patch theo thứ tự bản đồ giảm dần."""
    import torch
    import torch.nn.functional as F
    n = GRID * GRID
    order = torch.argsort(grid_map.flatten(), descending=True)
    ks = np.unique(np.round(np.linspace(0, n, n_steps)).astype(int))
    H, W = x.shape[-2:]
    masks = []
    for k in ks:
        m = torch.zeros(n, device=x.device)
        m[order[:k]] = 1.0                                  # patch được "giữ nổi bật" = top-k
        masks.append(F.interpolate(m.reshape(1, 1, GRID, GRID), size=(H, W), mode="nearest")[0])
    masks = torch.stack(masks)                              # (K,1,H,W)
    with torch.no_grad():
        p_del = F.softmax(model(x * (1 - masks)), -1)[:, cls].cpu().numpy()   # xoá top-k
        p_ins = F.softmax(model(x * masks), -1)[:, cls].cpu().numpy()         # chỉ giữ top-k
    return _auc(p_del), _auc(p_ins)


# ------------------------------------------------------------------ sanity check
def randomize_head_and_last_block(model, seed: int = 0):
    import torch
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    parts = [m.get_classifier()]
    parts.append(m.blocks[-1] if is_vit(m) else m.stages[-1])
    for part in parts:
        for mod in part.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    return m


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ------------------------------------------------------------------ điều phối
def run_xai(model, dataset, sample_idx, device, out_dir, log=print, test_top1=None, n_demo: int = N_DEMO):
    """dataset: đối tượng cho (tensor đã chuẩn hoá, nhãn) theo chỉ số 0..len-1; sample_idx: chỉ số trong dataset.
    Ghi out_dir/xai/{summary.json, per_image.npz, maps_demo.npz}."""
    import torch
    out = Path(out_dir) / "xai"
    out.mkdir(parents=True, exist_ok=True)
    model = model.float().to(device).eval()
    rand = randomize_head_and_last_block(model).float().to(device).eval()
    for p in list(model.parameters()) + list(rand.parameters()):
        p.requires_grad_(True)

    del_auc = {m: [] for m in METHODS}
    ins_auc = {m: [] for m in METHODS}
    sanity = {m: [] for m in METHODS}
    pred, gold, conf = [], [], []
    demo = {m: [] for m in METHODS}
    demo_rand = {m: [] for m in METHODS}
    demo_idx = []
    for n_done, i in enumerate(sample_idx):
        x, y = dataset[int(i)]
        x = x[None].to(device)
        with torch.no_grad():
            pr = torch.softmax(model(x), -1)[0]
        c = int(pr.argmax())
        pred.append(c); gold.append(int(y)); conf.append(float(pr[c]))
        maps = compute_maps(model, x, c)
        maps_r = compute_maps(rand, x, c)
        for m in METHODS:
            if m not in maps:
                continue
            d, ins = deletion_insertion(model, x, c, maps[m])
            del_auc[m].append(d); ins_auc[m].append(ins)
            sanity[m].append(spearman(maps[m].cpu().numpy(), maps_r[m].cpu().numpy()))
            if n_done < n_demo:
                demo[m].append(maps[m].cpu().numpy().astype(np.float16))
                demo_rand[m].append(maps_r[m].cpu().numpy().astype(np.float16))
        if n_done < n_demo:
            demo_idx.append(int(i))
        if (n_done + 1) % 25 == 0:
            log(f"[xai] {n_done + 1}/{len(sample_idx)}")

    rng = np.random.default_rng(0)

    def ci(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    summary = {"n_images": len(sample_idx), "sample_seed": SAMPLE_SEED, "grid": GRID, "n_steps": N_STEPS, "ig_steps": IG_STEPS,
               "explained_class": "predicted", "baseline": "mean image (0 in normalised space)",
               "pred_acc_on_sample": float(np.mean(np.array(pred) == np.array(gold))),
               "retrain_test_top1": test_top1, "methods": {}}
    for m in METHODS:
        if not del_auc[m]:
            continue
        summary["methods"][m] = {
            "deletion_auc_mean": float(np.mean(del_auc[m])), "deletion_auc_ci95": ci(del_auc[m]),
            "insertion_auc_mean": float(np.mean(ins_auc[m])), "insertion_auc_ci95": ci(ins_auc[m]),
            "sanity_spearman_mean": float(np.nanmean(sanity[m])), "sanity_spearman_ci95": ci(sanity[m])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    arr = {"sample_idx": np.asarray(sample_idx), "pred": np.array(pred), "gold": np.array(gold), "conf": np.array(conf)}
    for m in METHODS:
        if del_auc[m]:
            arr[f"{m}_del"] = np.array(del_auc[m]); arr[f"{m}_ins"] = np.array(ins_auc[m]); arr[f"{m}_sanity"] = np.array(sanity[m])
    np.savez(out / "per_image.npz", **arr)
    dm = {"idx": np.array(demo_idx)}
    for m in METHODS:
        if demo[m]:
            dm[m] = np.stack(demo[m]); dm[f"{m}_randomized"] = np.stack(demo_rand[m])
    np.savez(out / "maps_demo.npz", **dm)
    return summary
