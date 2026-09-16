"""Explicit held-out evaluation; never changes the selected checkpoint."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from settings import CACHE, RAW, RUNS
from training.model import load_adapter
from training.common import load_data, ranks, metrics, interval
from data.io import sha256


@torch.inference_mode()
def evaluate(checkpoint, features, reactions, split="test", proteins=None, pairs=None):
    torch.set_num_threads(4)
    data = load_data(features, reactions)
    model, cp = load_adapter(checkpoint)
    assert cp["encoder"] == data["encoder"], "Encoder contract changed"
    assert cp["teacher"] == data["teacher"], "Horizyn checkpoint changed"
    ids = [i for i, r in enumerate(data["records"]) if r["split"] == split]
    assert ids, f"No {split} records"
    rows = [data["records"][i] for i in ids]
    gold = data["gold"][ids]
    qgroups = data["groups"][gold].numpy()
    z = model(data["x"][ids])
    rr = ranks(z, data["library"], gold, data["groups"])
    styles = np.array([r["style"] for r in rows])
    report = dict(
        split=split,
        checkpoint_sha256=sha256(checkpoint),
        provenance=data["provenance"],
        reaction_groups=len(set(qgroups)),
        reaction_metrics=metrics(rr),
        recall_at_10_95ci=interval((rr <= 10).astype(float), qgroups),
        per_style={s: metrics(rr[styles == s]) for s in sorted(set(styles))},
        non_equation=metrics(rr[styles != "equation"])
        if (styles != "equation").any()
        else None,
    )
    if proteins is not None:
        pc = torch.load(proteins, map_location="cpu", weights_only=True)
        assert pc["teacher"] == data["teacher"]
        pv = F.normalize(pc["embeddings"].float(), dim=-1)
        pi = {p: i for i, p in enumerate(pc["ids"])}
        rc = torch.load(reactions, map_location="cpu", weights_only=True)
        ri = {r: i for i, r in enumerate(rc["ids"])}
        known = defaultdict(set)
        for path in pairs:
            with path.open() as f:
                for row in csv.DictReader(f):
                    if row["reaction_id"] in ri and row["protein_id"] in pi:
                        known[int(data["groups"][ri[row["reaction_id"]]])].add(
                            pi[row["protein_id"]]
                        )
        protein_ranks = []
        for start in range(0, len(z), 32):
            scores = z[start : start + 32] @ pv.T
            for j, g in enumerate(qgroups[start : start + len(scores)]):
                positives = list(known[int(g)])
                if positives:
                    protein_ranks.append(
                        1 + int((scores[j] > scores[j, positives].max()).sum())
                    )
        report["protein_metrics"] = metrics(protein_ranks) if protein_ranks else None
    report["limits"] = (
        "Adapter-held-out reaction groups, not unseen chemistry for Horizyn. Synthetic wording and incomplete protein labels limit interpretation."
    )
    return report, [{**r, "rank": int(rank)} for r, rank in zip(rows, rr)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=RUNS / "best/adapter.pt")
    p.add_argument("--features", type=Path, default=CACHE / "text.npz")
    p.add_argument("--reactions", type=Path, default=CACHE / "reactions.pt")
    p.add_argument(
        "--split", choices=["validation", "test", "diagnostic"], default="test"
    )
    p.add_argument("--proteins", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix(".ranks.jsonl").exists():
        raise SystemExit("Evaluation exists; choose a new output path")
    r, rows = evaluate(
        a.checkpoint,
        a.features,
        a.reactions,
        a.split,
        CACHE / "proteins.pt" if a.proteins else None,
        [RAW / "train_pairs.csv", RAW / "test_pairs.csv"],
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    from data.io import write_jsonl

    write_jsonl(a.out.with_suffix(".ranks.jsonl"), rows)
    print(json.dumps(r, indent=2))
