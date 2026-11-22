"""Do latency / throughput / bo nho / so tham so / GMACs cua mot model (regime bench, DEVIATIONS D10).

Khong dung du lieu that: anh ngau nhien 3x224x224. Model dung pretrained=False (trong so khong anh huong toc do).
"""
import time

import numpy as np


def _sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()


def _pct(xs_ms):
    a = np.asarray(xs_ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean()), "n": int(a.size)}


def _latency(model, x, device, amp, warmup, iters):
    import torch
    ts = []
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(x)
        _sync(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return _pct(ts)


def _throughput(model, device, amp, img, batch, warmup, iters, log):
    """img/s o batch lon; bat OOM thi ha doi batch va ghi lai."""
    import torch
    events = []
    b = batch
    x = None
    while b >= 1:
        try:
            x = torch.randn(b, 3, img, img, device=device)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                for _ in range(warmup):
                    model(x)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    model(x)
                _sync(device)
                dt = time.perf_counter() - t0
            return {"batch_requested": batch, "batch_used": b, "img_per_s": b * iters / dt, "iters": iters,
                    "oom_events": events}
        except torch.cuda.OutOfMemoryError as e:
            events.append({"batch": b, "error": str(e)[:200]})
            log(f"[bench] OOM o batch {b}, ha xuong {b // 2}")
            x = None
            torch.cuda.empty_cache()
            b //= 2
    return {"batch_requested": batch, "batch_used": None, "img_per_s": None, "iters": iters, "oom_events": events}


def gmacs(model, img, device):
    """GMACs 1 anh = FLOPs/2 theo FlopCounterMode (chi dem matmul/conv/attention)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, img, img, device=device)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9


def measure(model, device, img, smoke, log):
    import torch
    model = model.to(device).eval()
    warmup, iters = (2, 5) if smoke else (50, 200)
    tp_batch, tp_warm, tp_iters = (8, 1, 3) if smoke else (64, 5, 30)
    out = {"image_size": img, "warmup": warmup, "iters": iters, "device": device,
           "params": int(sum(p.numel() for p in model.parameters())),
           "gmacs": gmacs(model, img, device)}
    x1 = torch.randn(1, 3, img, img, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        out["weights_mb"] = torch.cuda.memory_allocated() / 2 ** 20
    out["latency_b1_fp32"] = _latency(model, x1, device, False, warmup, iters)
    if device == "cuda":
        out["latency_b1_fp16_autocast"] = _latency(model, x1, device, True, warmup, iters)
        out["peak_mem_latency_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        torch.cuda.reset_peak_memory_stats()
        out["throughput_fp16_autocast"] = _throughput(model, device, True, img, tp_batch, tp_warm, tp_iters, log)
        out["peak_mem_throughput_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    else:   # CPU: khong co fp16 autocast; do fp32 batch nho (ghi ro de khong nham voi GPU)
        out["throughput_fp32_cpu"] = _throughput(model, device, False, img, 8 if smoke else 16, tp_warm, tp_iters, log)
    return out
