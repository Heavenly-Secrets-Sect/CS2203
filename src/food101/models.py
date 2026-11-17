import json


def build_model(mcfg, num_classes, img_size, smoke=False):
    import timm
    if smoke:  # model tiny, khong tai trong so
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes,
                                 img_size=img_size, depth=2)
    kw = {"img_size": img_size} if mcfg.get("pass_img_size") else {}
    return timm.create_model(mcfg["timm_id"], pretrained=True, num_classes=num_classes, **kw)


def resolved_revision(mcfg, smoke):
    """sha repo HF luc chay (best effort, co the None neu khong co mang)."""
    if smoke:
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(mcfg["hf_id"], timeout=15).sha
    except Exception:
        return None


def features_fn(model):
    """Dac trung truoc head (pre_logits) - dung cho LP."""
    def f(x):
        return model.forward_head(model.forward_features(x), pre_logits=True)
    return f


def head_module(model):
    return model.get_classifier()


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))
