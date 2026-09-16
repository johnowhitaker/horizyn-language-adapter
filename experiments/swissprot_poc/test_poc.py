"""Regression checks for leakage, retrieval ties and instruction-specific demo caches."""
import unittest
from unittest.mock import Mock, patch
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from fastapi import HTTPException
from .common import clean_function, digest
from .prepare import components, split_for
from .learning import retrieval_ranks
from . import learning
from data.io import write_jsonl, sha256
from training.model import load_adapter
from web_demo.server import Engine, Query


class DataTests(unittest.TestCase):
    def test_duplicate_text_joins_distant_sequence_clusters(self):
        rows = [dict(protein_id=p, function=t, sequence_sha256=digest(p)) for p, t in
                [("a", "Hydrolyzes ATP."), ("b", "Hydrolyzes ATP!"), ("c", "Makes something else.")]]
        groups = components(rows, [("a", "a"), ("b", "c")])
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(len({split_for(g) for g in groups.values()}), 1)

    def test_function_cleanup_retains_biology(self):
        text = "FUNCTION: Hydrolyzes ATP (By similarity). {ECO:0000250}. Uses Mg(2+) (PubMed:123)."
        cleaned = clean_function(text)
        self.assertNotIn("ECO:", cleaned)
        self.assertNotIn("PubMed", cleaned)
        self.assertIn("Hydrolyzes ATP", cleaned)
        self.assertIn("Mg(2+)", cleaned)

    def test_rank_ties_do_not_turn_constant_predictions_into_success(self):
        z = torch.zeros(1, 2)
        lib = torch.randn(100, 2)
        np.testing.assert_allclose(retrieval_ranks(z, lib, [[0]]), [50.5])
        np.testing.assert_allclose(retrieval_ranks(z, lib, [[0, 1]]), [50.0])

    def test_best_positive_retrieval(self):
        z = torch.tensor([[1., 0.]])
        lib = torch.tensor([[.1, .9], [.4, .6], [.9, .1]])
        np.testing.assert_allclose(retrieval_ranks(z, lib, [[0, 2]]), [1])

    def test_training_round_trip_selects_a_usable_checkpoint(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, features = root / "dataset", root / "features"
            dataset.mkdir()
            features.mkdir()
            rows = [dict(protein_id=str(i), group_id=str(i), text_group=str(i),
                split="train" if i < 60 else "validation", text=f"enzyme {i}", style="name_function") for i in range(80)]
            write_jsonl(dataset / "texts.jsonl", rows)
            write_jsonl(dataset / "proteins.jsonl", rows)
            x = rng.normal(size=(80, 8)).astype("float32")
            y = torch.from_numpy(x @ rng.normal(size=(8, 512)).astype("float32"))
            torch.save(dict(ids=[r["protein_id"] for r in rows], embeddings=y,
                groups=[r["group_id"] for r in rows], text_groups=[r["text_group"] for r in rows],
                teacher={"fixture": True}), dataset / "targets.pt")
            manifest = {key + "_sha256": sha256(dataset / name) for key, name in
                        [("texts", "texts.jsonl"), ("proteins", "proteins.jsonl"), ("targets", "targets.pt")]}
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            meta = dict(encoder={"fixture": True}, records=rows, source_sha256=manifest["texts_sha256"])
            np.savez(features / "protein.npz", embeddings=x, metadata=json.dumps(meta))
            with patch.object(learning, "ARTIFACTS", root), contextlib.redirect_stdout(io.StringIO()):
                learning.train("cpu")
            model, checkpoint = load_adapter(root / "selected/adapter.pt")
            self.assertEqual(checkpoint["target_type"], "protein")
            self.assertTrue(torch.isfinite(model(torch.from_numpy(x))).all())
            self.assertGreater(checkpoint["validation"]["recall_at_10"], .9)


class DemoTests(unittest.TestCase):
    def engine(self):
        e = Engine.__new__(Engine)
        e.query_cache = {}
        e.checkpoint = {"encoder": {"prefix": "reaction: "}}
        e.protein_checkpoint = {"encoder": {"prefix": "protein: "}}
        e.encoder = Mock(side_effect=lambda texts, prefix: torch.tensor(
            [[1., 0.] if prefix == "reaction: " else [0., 1.] for _ in texts]))
        e.adapter = torch.nn.Identity()
        e.protein_adapter = torch.nn.Identity()
        e.reactions = e.proteins = torch.eye(2)
        e.ids = ["Rh_1", "Rh_2"]
        e.pids = ["P1", "P2"]
        e.ri = {p: i for i, p in enumerate(e.ids)}
        e.pi = {p: i for i, p in enumerate(e.pids)}
        e.groups = e.ids
        e.metadata = [{}, {}]
        e.protein_meta = {p: {"name": p} for p in e.pids}
        return e

    def test_mode_switch_uses_distinct_features_and_reuses_each_cache(self):
        e = self.engine()
        a = e.search("same text", "adapter", "P2")
        p = e.search("same text", "protein", "P2")
        e.search("same text", "adapter")
        self.assertEqual(e.encoder.call_count, 2)
        self.assertEqual(a["proteins"][0]["id"], "P1")
        self.assertEqual(p["proteins"][0]["id"], "P2")
        self.assertEqual(a["expected"]["rank"], 2)
        self.assertEqual(p["expected"]["rank"], 1)
        self.assertEqual(p["expected"]["kind"], "protein")

    def test_optional_adapter_unavailable_is_explicit(self):
        e = self.engine()
        e.protein_adapter = None
        with self.assertRaises(HTTPException) as raised:
            e.representation(["ATP"], "protein")
        self.assertEqual(raised.exception.status_code, 400)
        e.encoder.assert_not_called()
        self.assertEqual(Query(text="ATP", mode="protein").mode, "protein")


if __name__ == "__main__":
    unittest.main()
