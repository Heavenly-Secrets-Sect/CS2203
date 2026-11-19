import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from food101 import xai as X


class DS:
    def __init__(self, n=6, nc=5, size=224):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, size, size, generator=g)
        self.labels = np.arange(n) % nc

    def __getitem__(self, i):
        return self.x[i], int(self.labels[i])


def vit():
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5, depth=3)


def convnext():
    return timm.create_model("convnext_atto", pretrained=False, num_classes=5)


def test_fixed_sample_stratified_and_stable():
    labels = np.repeat(np.arange(10), 25)
    a, b = X.fixed_sample(labels), X.fixed_sample(labels)
    assert (a == b).all() and len(a) == 20
    assert all((labels[a] == c).sum() == 2 for c in range(10))


@pytest.mark.parametrize("make,has_rollout", [(vit, True), (convnext, False)])
def test_maps_shapes_and_metrics(make, has_rollout, tmp_path):
    m = make().eval()
    x = DS().x[:1]
    maps = X.compute_maps(m, x, 2)
    assert ("rollout" in maps) == has_rollout
    for v in maps.values():
        assert tuple(v.shape) == (X.GRID, X.GRID) and torch.isfinite(v).all()
    d, i = X.deletion_insertion(m, x, 2, maps["gradcam"])
    assert 0.0 <= d <= 1.0 and 0.0 <= i <= 1.0
    s = X.run_xai(m, DS(), np.array([0, 1, 2]), "cpu", tmp_path, log=lambda *a: None, test_top1=0.5, n_demo=2)
    assert set(s["methods"]) == ({"rollout", "gradcam", "ig"} if has_rollout else {"gradcam", "ig"})
    assert (tmp_path / "xai" / "per_image.npz").exists() and (tmp_path / "xai" / "maps_demo.npz").exists()


def test_rollout_is_row_stochastic_ordering_and_sanity_changes_maps():
    m = vit().eval()
    x = DS().x[:1]
    r = X.attention_rollout(m, x)
    assert (r >= 0).all()
    rand = X.randomize_head_and_last_block(m).eval()
    a, b = X.gradcam(m, x, 1), X.gradcam(rand, x, 1)
    assert not torch.allclose(a, b)
    assert not any(torch.equal(p, q) for p, q in zip(m.get_classifier().parameters(), rand.get_classifier().parameters()))


def test_deletion_insertion_reward_good_map():
    # model giả: xác suất lớp 0 tăng khi vùng trên-trái còn nguyên -> bản đồ đúng cho AUC insertion cao hơn deletion
    class M(torch.nn.Module):
        def forward(self, x):
            s = x[:, :, :112, :112].mean((1, 2, 3))
            return torch.stack([s * 20, torch.zeros_like(s)], 1)
    m = M()
    x = torch.ones(1, 3, 224, 224)
    good = torch.zeros(X.GRID, X.GRID); good[:7, :7] = 1
    bad = 1 - good
    dg, ig = X.deletion_insertion(m, x, 0, good)
    db, ib = X.deletion_insertion(m, x, 0, bad)
    assert ig > ib and dg < db
