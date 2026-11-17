import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import features_fn, head_module


def _worker_init(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    import random
    random.seed(s)


def make_loader(ds, batch_size, shuffle, seed, num_workers, drop_last=False, device="cpu"):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, generator=g,
                      num_workers=num_workers, worker_init_fn=_worker_init if num_workers else None,
                      persistent_workers=bool(num_workers and shuffle), pin_memory=(device == "cuda"))


def _autocast(device, amp):
    return torch.autocast(device_type=device, dtype=torch.float16, enabled=bool(amp and device == "cuda"))


@torch.no_grad()
def predict_logits(model, loader, device, amp):
    model.eval()
    out = []
    for x, _ in loader:
        with _autocast(device, amp):
            y = model(x.to(device, non_blocking=True))
        out.append(y.float().cpu())
    return torch.cat(out).numpy()


def _param_groups(model, wd):
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodec if p.ndim < 2 or n in skip else dec).append(p)
    return [{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}]


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def finetune(model, train_ds, dev_loader, cfg_t, lr, micro_batch, seed, num_workers, device,
             epochs, max_steps=None, log=print):
    """AdamW + cosine (warmup 10%) + label smoothing, fp16 AMP + GradScaler, chon epoch cuoi."""
    model.to(device)
    accum = max(1, int(round(cfg_t["batch_size"] / micro_batch)))
    loader = make_loader(train_ds, micro_batch, True, seed, num_workers, drop_last=True, device=device)
    steps_per_epoch = len(loader) // accum
    if steps_per_epoch < 1:
        raise RuntimeError(f"Tap train ({len(train_ds)}) nho hon 1 batch hieu dung ({micro_batch}x{accum})")
    total = steps_per_epoch * epochs
    warmup = int(round(cfg_t["warmup_frac"] * total))
    opt = torch.optim.AdamW(_param_groups(model, cfg_t["weight_decay"]), lr=lr)
    use_amp = bool(cfg_t["amp"] and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ls = cfg_t["label_smoothing"]
    clip = cfg_t.get("grad_clip") or 0.0
    hist, step, nan_steps, t_train = [], 0, 0, 0.0
    truncated = False
    for ep in range(epochs):
        model.train()
        t0, seen, loss_sum, n_loss = time.time(), 0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader):
            if i >= steps_per_epoch * accum:
                break
            if i % accum == 0:
                for gp in opt.param_groups:
                    gp["lr"] = lr_at(step, total, warmup, lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _autocast(device, cfg_t["amp"]):
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=ls)
            if not torch.isfinite(loss):
                nan_steps += 1
            scaler.scale(loss / accum).backward()
            seen += len(x)
            loss_sum += float(loss.detach()) if torch.isfinite(loss) else 0.0
            n_loss += 1
            if (i + 1) % accum == 0:
                if clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if max_steps is not None and step >= max_steps:
                    truncated = True
                    break
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        rec = {"epoch": ep + 1, "train_loss": loss_sum / max(1, n_loss), "lr_end": opt.param_groups[0]["lr"],
               "epoch_time_s": dt, "train_img_per_s": seen / max(dt, 1e-9), "steps": step}
        if dev_loader is not None:
            lg = predict_logits(model, dev_loader, device, cfg_t["amp"])
            rec["dev_acc"] = float((lg.argmax(1) == dev_loader.dataset.labels).mean())
        hist.append(rec)
        log(f"[ft] {rec}")
        if truncated:
            break
    return {"epochs": hist, "nan_loss_steps": nan_steps, "truncated_by_max_steps": truncated,
            "accum": accum, "micro_batch": micro_batch, "steps_per_epoch": steps_per_epoch,
            "total_steps": total, "train_time_s": t_train}


# ---------- linear probe ----------
@torch.no_grad()
def extract_features(model, loader, device, amp):
    model.eval().to(device)
    f = features_fn(model)
    out = []
    for x, _ in loader:
        with _autocast(device, amp):
            out.append(f(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def fit_linear(ftr, ytr, fdev, ydev, cfg_lp, wd_grid, seed, device, log=print):
    """Train head tren dac trung da chuan hoa; chon weight decay (L2) theo dev acc.
    Tra ve (W, b) da gop chuan hoa (dung truc tiep tren dac trung goc) + grid log."""
    mean, std = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xdev = ((ftr - mean) / std).to(device), ((fdev - mean) / std).to(device)
    ytr_t, ydev_t = torch.as_tensor(ytr).to(device), torch.as_tensor(ydev).to(device)
    nc = int(max(ytr.max(), ydev.max())) + 1
    grid, best = [], None
    for wd in wd_grid:
        g = torch.Generator().manual_seed(seed)
        lin = torch.nn.Linear(xtr.shape[1], nc).to(device)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        opt = torch.optim.Adam(lin.parameters(), lr=cfg_lp["lr"])
        n, bs = len(xtr), cfg_lp["batch_size"]
        steps = cfg_lp["epochs"] * math.ceil(n / bs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(cfg_lp["epochs"]):
            perm = torch.randperm(n, generator=g).to(device)
            for k in range(0, n, bs):
                idx = perm[k:k + bs]
                loss = F.cross_entropy(lin(xtr[idx]), ytr_t[idx]) + 0.5 * wd * (lin.weight ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        with torch.no_grad():
            acc = float((lin(xdev).argmax(1) == ydev_t).float().mean())
        grid.append({"weight_decay": wd, "dev_acc": acc})
        log(f"[lp] wd={wd} dev_acc={acc:.4f}")
        if best is None or acc > best[0]:  # hoa: giu wd nho hon (xuat hien truoc)
            best = (acc, wd, lin.weight.detach().cpu().clone(), lin.bias.detach().cpu().clone())
    acc, wd, W, b = best
    Wf = W / std                                   # W((x-m)/s)+b = (W/s)x + (b - (W/s)m)
    bf = b - (Wf * mean).sum(1)
    return Wf, bf, {"grid": grid, "chosen_weight_decay": wd, "chosen_dev_acc": acc}


def set_head(model, W, b):
    h = head_module(model)
    with torch.no_grad():
        h.weight.copy_(W)
        h.bias.copy_(b)
