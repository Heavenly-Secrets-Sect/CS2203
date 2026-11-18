"""python -m food101.run_job --model KEY --regime {ft,lp,lpft,zeroshot,sweep} --seed N --out DIR [...]"""
import argparse
import copy
import json
import os
import resource
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import data as D
from . import evaluate as E
from . import splits as S
from .utils import (git_commit, lib_versions, load_configs, seed_everything, splits_dir, write_json,
                    write_yaml)

REGIMES = ("ft", "lp", "lpft", "zeroshot", "sweep")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="food101.run_job")
    p.add_argument("--model", required=True)
    p.add_argument("--regime", required=True, choices=REGIMES)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=None, help="LR dinh (batch 128); mac dinh lay tu configs/models.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="dung sau N optimizer step (do throughput/smoke CI)")
    p.add_argument("--smoke", action="store_true", help="du lieu gia + model tiny, khong mang, chay CPU")
    # co moi, mac dinh khong doi hanh vi
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--no-check-counts", action="store_true")
    p.add_argument("--skip-if-done", action="store_true", help="bo qua neu DIR da co file DONE")
    return p.parse_args(argv)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _num_workers(cfg, smoke):
    if smoke:
        return 0
    if cfg["train"].get("num_workers") is not None:
        return int(cfg["train"]["num_workers"])
    env = os.environ.get("FOOD101_NUM_WORKERS")
    return int(env) if env else min(4, os.cpu_count() or 1)


def _peak_mem(device):
    import torch
    if device == "cuda":
        return {"cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                "cuda_max_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20}
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"cpu_max_rss_mb": r / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)}


def run(args):
    import torch
    t_start = time.time()
    out = Path(args.out)
    if args.skip_if_done and (out / "DONE").exists():
        log(f"[run] {out} da DONE, bo qua")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)

    cfg, models = load_configs()
    cfg = copy.deepcopy(cfg)
    if args.model not in models:
        raise SystemExit(f"model '{args.model}' khong co trong models.yaml: {sorted(models)}")
    mcfg = models[args.model]
    regime, smoke = args.regime, args.smoke
    device = "cuda" if torch.cuda.is_available() and not smoke else "cpu"
    seed_everything(args.seed)

    # ---- hyperparameters
    tcfg = cfg["train"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    frac = args.train_frac if args.train_frac is not None else 1.0
    if regime == "sweep":
        epochs = args.epochs if args.epochs is not None else cfg["sweep"]["epochs"]
        frac = args.train_frac if args.train_frac is not None else cfg["sweep"]["train_frac"]
    img_size = mcfg["img_size"]
    resize = cfg["test"]["resize"]
    micro_batch = args.micro_batch or mcfg["micro_batch"]
    if smoke:
        img_size, resize = 32, 37
        tcfg["batch_size"] = 32
        micro_batch = args.micro_batch or 16
    lr = args.lr if args.lr is not None else mcfg.get("lr")
    if regime in ("ft", "lpft", "sweep") and lr is None:
        if smoke:
            lr = 5e-4
        else:
            raise SystemExit(f"model '{args.model}' chua co lr (chay sweep roi dien configs/models.yaml, hoac truyen --lr)")
    if regime in ("sweep",) and args.lr is None and not smoke:
        raise SystemExit("regime sweep can --lr")

    # ---- data
    if smoke:
        tmp = tempfile.mkdtemp(prefix="food101_smoke_")
        meta = D.build_smoke_cache(cfg, tmp, short=resize)
    else:
        meta = D.ensure_cache(cfg)
        if not args.no_check_counts:
            D.check_counts(meta, cfg)
    sdir = meta["dir"]
    class_names = meta["class_names"]
    nc = len(class_names)
    train_labels = np.load(Path(sdir) / "train.labels.npy")
    train_idx, dev_idx, split_hash = S.load_or_create(
        train_labels, splits_dir(), cfg["split"]["dev_frac"], cfg["split"]["seed"], write=not smoke)
    fit_idx = S.subsample_stratified(train_idx, train_labels, frac, cfg["split"]["sweep_subset_seed"])
    tf_train, tf_test = D.make_transforms(mcfg, img_size, resize, tcfg["rrc_scale"])
    nw = _num_workers(cfg, smoke)
    from .train import (extract_features, finetune, fit_linear, make_loader, predict_logits, set_head)
    ebs = micro_batch * 2

    def eval_loader(name, idx):
        ds = D.ShardDataset(sdir, name, idx, tf_test)
        return ds, make_loader(ds, ebs, False, 0, nw, device=device)

    dev_ds, dev_loader = eval_loader("train", dev_idx)
    do_test = regime != "sweep"
    test_idx = np.arange(meta["splits"]["validation"]["n"])
    test_ds, test_loader = eval_loader("validation", test_idx) if do_test else (None, None)
    log(f"[run] model={args.model} regime={regime} seed={args.seed} device={device} "
        f"fit={len(fit_idx)} dev={len(dev_idx)} test={len(test_idx) if do_test else 0} lr={lr} epochs={epochs}")

    # ---- model
    from .models import build_model, count_params, resolved_revision
    history = {"regime": regime}
    extra = {}
    model = None
    t_train0 = time.time()

    if regime == "zeroshot":
        dev_logits, test_logits, extra = zeroshot(mcfg, cfg, class_names, dev_loader, test_loader,
                                                  device, smoke, img_size)
    else:
        model = build_model(mcfg, nc, img_size, smoke)
        n_params = count_params(model)
        extra["params"] = n_params
        if regime in ("lp", "lpft"):
            fit_ds = D.ShardDataset(sdir, "train", fit_idx, tf_test)  # dac trung cache 1 lan, khong augment
            ftr = extract_features(model, make_loader(fit_ds, ebs, False, 0, nw, device=device), device, tcfg["amp"])
            fdv = extract_features(model, make_loader(dev_ds, ebs, False, 0, nw, device=device), device, tcfg["amp"])
            W, b, lp_info = fit_linear(ftr, fit_ds.labels, fdv, dev_ds.labels, cfg["lp"],
                                       cfg["lp"]["weight_decay_grid"], args.seed, device, log)
            history["lp"] = lp_info
            set_head(model, W, b)
            if regime == "lp":
                for p in model.parameters():
                    p.requires_grad_(False)
        if regime in ("ft", "lpft", "sweep"):
            fit_ds = D.ShardDataset(sdir, "train", fit_idx, tf_train)
            ft_hist = finetune(model, fit_ds, dev_loader, tcfg, lr, micro_batch, args.seed, nw, device,
                               epochs, args.max_steps, log)
            history["finetune"] = ft_hist
        model.to(device)
        dev_logits = predict_logits(model, dev_loader, device, tcfg["amp"])
        test_logits = predict_logits(model, test_loader, device, tcfg["amp"]) if do_test else None
        if args.save_ckpt:
            import torch
            torch.save(model.state_dict(), out / "checkpoint.pt")
    t_train = time.time() - t_train0

    # ---- ghi output (logits float16 -> scorer tinh lai TU FILE)
    np.save(out / "dev_logits.npy", dev_logits.astype(np.float16))
    metrics = {"regime": regime, "model": args.model, "seed": args.seed,
               "dev": E.score_files(out / "dev_logits.npy", dev_ds.labels, cfg["metrics"]["ece_bins"])}
    if do_test:
        np.save(out / "test_logits.npy", test_logits.astype(np.float16))
        gold = test_ds.labels
        ids = [f"validation/{int(i)}" for i in test_idx]
        E.write_predictions(out / "test_predictions.jsonl", ids, gold,
                            np.load(out / "test_logits.npy").astype(np.float32))
        metrics["test"] = E.score_files(out / "test_logits.npy", gold, cfg["metrics"]["ece_bins"])
    write_json(out / "metrics.json", metrics)
    write_json(out / "history.json", history)

    if smoke:
        shutil.rmtree(sdir, ignore_errors=True)
    run_cfg = {
        "command": " ".join(getattr(sys, "orig_argv", sys.argv)), "argv": sys.argv[1:], "git_commit": git_commit(),
        "versions": lib_versions(),
        "gpu": _gpu_name(device), "device": device, "smoke": smoke,
        "model": {"key": args.model, "timm_id": mcfg["timm_id"], "hf_id": mcfg["hf_id"],
                  "revision_checked": mcfg.get("revision_checked"),
                  "revision_resolved": resolved_revision(mcfg, smoke),
                  "mean": mcfg["mean"], "std": mcfg["std"], "interpolation": mcfg["interpolation"],
                  "img_size": img_size, **extra},
        "regime": regime, "seed": args.seed,
        "hyperparameters": {"lr": lr, "lr_source": "cli" if args.lr is not None else "models.yaml",
                            "epochs": epochs, "train_frac": frac, "max_steps": args.max_steps,
                            "micro_batch": micro_batch, "effective_batch": tcfg["batch_size"],
                            "weight_decay": tcfg["weight_decay"], "warmup_frac": tcfg["warmup_frac"],
                            "label_smoothing": tcfg["label_smoothing"], "rrc_scale": tcfg["rrc_scale"],
                            "grad_clip": tcfg.get("grad_clip"), "amp_fp16": bool(tcfg["amp"] and device == "cuda"),
                            "test_resize": resize, "num_workers": nw,
                            "lp": cfg["lp"] if regime in ("lp", "lpft") else None,
                            "lp_chosen": history.get("lp", {}).get("chosen_weight_decay")},
        "data": {"dataset": cfg["dataset"]["name"] if not smoke else "SMOKE-fake",
                 "dataset_revision": cfg["dataset"].get("revision") if not smoke else None,
                 "n_fit": len(fit_idx), "n_dev": len(dev_idx), "n_test": len(test_idx) if do_test else 0,
                 "split_seed": cfg["split"]["seed"], "dev_ids_hash": split_hash,
                 "cache_meta": {k: meta["splits"][k] for k in meta["splits"]},
                 "test_equals_hf_validation_unverified": True},
        "time": {"start_unix": t_start, "total_s": time.time() - t_start, "fit_and_eval_s": t_train},
        "peak_memory": _peak_mem(device),
        "status": "ok",
    }
    write_yaml(out / "run_config.yaml", run_cfg)
    (out / "DONE").write_text("ok\n")
    return out


def _gpu_name(device):
    import torch
    return torch.cuda.get_device_name(0) if device == "cuda" else None


def zeroshot(mcfg, cfg, class_names, dev_loader, test_loader, device, smoke, img_size):
    import torch
    import torch.nn.functional as F
    prompts = [cfg["zeroshot"]["prompt"].format(n.replace("_", " ")) for n in class_names]
    if smoke:  # stub: KHONG phai CLIP. Chi de kiem thu duong ong.
        from .models import build_model
        enc = build_model(mcfg, 16, img_size, True).eval()
        g = torch.Generator().manual_seed(0)
        T = F.normalize(torch.randn(len(prompts), 16, generator=g), dim=-1)
        enc_img = lambda x: enc(x)  # noqa: E731
        scale = 100.0
        info = {"zeroshot_impl": "SMOKE-STUB"}
    else:
        if "open_clip" not in mcfg:
            raise SystemExit(f"model '{mcfg['timm_id']}' khong co cau hinh open_clip; zeroshot chi cho clip_base")
        import open_clip  # lazy
        oc = mcfg["open_clip"]
        m, _, _ = open_clip.create_model_and_transforms(oc["arch"], pretrained=oc["pretrained"])
        m = m.to(device).eval()
        tok = open_clip.get_tokenizer(oc["arch"])
        with torch.no_grad():
            T = F.normalize(m.encode_text(tok(prompts).to(device)).float(), dim=-1).cpu()
        enc_img = m.encode_image
        scale = float(m.logit_scale.detach().exp())
        info = {"zeroshot_impl": "open_clip", "open_clip": oc, "prompt": cfg["zeroshot"]["prompt"],
                "logit_scale": scale}

    @torch.no_grad()
    def run_loader(loader):
        out = []
        for x, _ in loader:
            f = F.normalize(enc_img(x.to(device)).float(), dim=-1).cpu()
            out.append(scale * f @ T.T)
        return torch.cat(out).numpy()

    return run_loader(dev_loader), run_loader(test_loader), info


def main(argv=None):
    args = parse_args(argv)
    try:
        out = run(args)
    except SystemExit as e:
        if e.code not in (0, None):
            print(json.dumps({"status": "error", "error": str(e.code)}))
            raise SystemExit(1)
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        print(json.dumps({"status": "error", "error": f"{type(e).__name__}: {e}"}))
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "dir": str(out)}))


if __name__ == "__main__":
    main()
