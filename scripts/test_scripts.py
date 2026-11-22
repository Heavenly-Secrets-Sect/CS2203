"""Test nhỏ cho plan.py, audit_runs.py, stats.py bằng dữ liệu giả. Chạy: python3 -m unittest scripts/test_scripts.py -v"""
import json, shutil, sys, tempfile, unittest, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_runs, plan, stats  # noqa: E402

NT, ND, NC = 40, 12, 101


def write_job(root: Path, job: dict, rng, *, corrupt=None):
    d = root / "outputs" / job["out_rel"]
    d.mkdir(parents=True, exist_ok=True)
    y = np.arange(NT) % NC
    lg = rng.normal(size=(NT, NC)).astype("float32")
    lg[np.arange(NT), y] += 3.0
    lg[::4, :] = rng.normal(size=(len(lg[::4]), NC))
    pred = lg.argmax(1)
    acc = float((pred == y).mean())
    (d / "metrics.json").write_text(json.dumps({"test": {"top1": acc}, "dev": {"top1": 0.5}}))
    (d / "history.json").write_text(json.dumps([{"epoch": 1, "loss": 1.0}]))
    (d / "run_config.yaml").write_text("a: 1\n")
    np.save(d / "test_logits.npy", lg); np.save(d / "dev_logits.npy", rng.normal(size=(ND, NC)).astype("float32"))
    rows = [{"idx": i, "label": int(y[i]), "pred": int(pred[i])} for i in range(NT)]
    if corrupt == "label":
        rows[3]["label"] = 101
    if corrupt == "acc":
        (d / "metrics.json").write_text(json.dumps({"test": {"top1": acc + 0.2}}))
    if corrupt == "nan":
        lg[0, 0] = np.nan; np.save(d / "test_logits.npy", lg)
    if corrupt == "short":
        rows = rows[:-2]
    (d / "test_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class PlanTests(unittest.TestCase):
    def test_deterministic_and_consistent(self):
        p = json.loads(plan.PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(plan.plan_hash(p), p["plan_hash"])
        for s in p["stages"].values():
            ids = [j["id"] for j in s["jobs"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(s["n_sessions"], len(s["sessions"]))
        self.assertEqual(len(p["stages"]["sweep"]["jobs"]), 27)
        self.assertEqual(len(p["stages"]["p0a"]["jobs"]), 16)
        self.assertEqual(len(p["stages"]["p0b"]["jobs"]), 13)
        self.assertEqual(len(p["stages"]["explore"]["jobs"]), 6)
        l = [j for j in p["stages"]["p0a"]["jobs"] if j["model"] == "vit_large" and j["regime"] != "lp"]
        self.assertEqual(sorted((j["regime"], j["seed"]) for j in l), [("ft", 42), ("ft", 123), ("ft", 2024), ("lpft", 42)])
        for sess in p["stages"]["p0a"]["sessions"]:
            models = {i.split("__")[0] for lane in sess["lanes"] for i in lane}
            self.assertEqual("vit_large" in models, sess["group"] == "vit_large")
            if sess["group"] == "vit_large":
                self.assertEqual(models, {"vit_large"})

    def test_config_hash_stable(self):
        j = plan._job("vit_base", "ft", 42, lr="selected")
        self.assertEqual(plan.config_hash(j), plan.config_hash(dict(j)))
        self.assertNotEqual(plan.config_hash(j), plan.config_hash(plan._job("vit_base", "ft", 123, lr="selected")))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = json.loads(plan.PLAN_PATH.read_text(encoding="utf-8"))
        self.jobs = self.plan["stages"]["explore"]["jobs"]
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_audit(self):
        return audit_runs.audit(self.plan, [self.tmp], ["explore"], NT, ND, NC)

    def fill(self, **bad):
        for j in self.jobs:
            write_job(self.tmp, j, self.rng, corrupt=bad.get(j["id"]))

    def test_clean(self):
        self.fill()
        r = self.run_audit()
        self.assertTrue(r["ok"], r["problems"])

    def test_missing_and_extra(self):
        self.fill()
        shutil.rmtree(self.tmp / "outputs" / self.jobs[0]["out_rel"])
        (self.tmp / "outputs" / "foo_ft" / "seed_1").mkdir(parents=True)
        (self.tmp / "outputs" / "foo_ft" / "seed_1" / "metrics.json").write_text("{}")
        r = self.run_audit()
        self.assertTrue(any("THIẾU" in p for p in r["problems"]))
        self.assertTrue(any("THỪA" in p for p in r["problems"]))

    def test_corruptions(self):
        for kind, needle in (("label", "ngoài 0..100"), ("acc", "acc metrics.json"), ("nan", "NaN"), ("short", "dòng")):
            with self.subTest(kind):
                shutil.rmtree(self.tmp / "outputs", ignore_errors=True)
                self.fill(**{self.jobs[0]["id"]: kind})
                r = self.run_audit()
                self.assertFalse(r["ok"])
                self.assertTrue(any(needle in p for p in r["problems"]), r["problems"])

    def test_zip_input(self):
        self.fill()
        z = self.tmp / "a.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for f in (self.tmp / "outputs").rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(self.tmp).as_posix())
        with tempfile.TemporaryDirectory() as t:
            root = audit_runs.extract(z, Path(t))
            self.assertTrue(audit_runs.audit(self.plan, [root], ["explore"], NT, ND, NC)["ok"])


class StatsTests(unittest.TestCase):
    def test_score_and_holm(self):
        y = np.arange(50) % 5
        lg = np.eye(5)[y] * 10.0
        s = stats.score(lg, y)
        self.assertEqual(s["top1"], 1.0); self.assertEqual(s["macro_f1"], 1.0); self.assertLess(s["nll"], 1e-3)
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        b = stats.paired_bootstrap(lg, lg, y, B=200)
        self.assertEqual(b["diff"], 0.0)



class ScorePrecisionRecallTest(unittest.TestCase):
    def test_macro_precision_recall_balanced(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        lg = np.eye(3)[np.array([0, 1, 1, 1, 2, 0])] * 5.0     # dự đoán: 0,1,1,1,2,0
        s = stats.score(lg, y)
        self.assertAlmostEqual(s["top1"], 4 / 6)
        self.assertAlmostEqual(s["macro_recall"], s["top1"])       # lớp cân bằng: macro recall = top-1
        self.assertAlmostEqual(s["macro_precision"], (1 / 2 + 2 / 3 + 1) / 3)


class PackXaiTest(unittest.TestCase):
    def test_only_fp16_checkpoint_allowed(self):
        import pack_runs
        from pathlib import Path as P
        self.assertTrue(pack_runs.wanted(P("outputs/vit_base_ft-xai/seed_42/model_fp16.pt")))
        self.assertTrue(pack_runs.wanted(P("outputs/vit_base_ft-xai/seed_42/xai/per_image.npz")))
        self.assertFalse(pack_runs.wanted(P("outputs/vit_base_ft/seed_42/checkpoint.pt")))
        self.assertFalse(pack_runs.wanted(P("outputs/x/model.safetensors")))

    def test_xai_stage_plan(self):
        p = plan.build_plan(plan.load_estimates())
        st = p["stages"]["xai"]
        self.assertEqual(st["n_jobs"], 3)
        self.assertTrue(all(j.get("xai") and j["seed"] == 42 and j["lr"] == "selected" for j in st["jobs"]))
        self.assertEqual(st["n_sessions"], 2)


if __name__ == "__main__":
    unittest.main()
