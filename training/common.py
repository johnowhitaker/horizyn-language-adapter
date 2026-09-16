"""Artifact joins and metrics shared by training, evaluation and the demo."""

import json
import numpy as np
import torch
from torch.nn import functional as F
from data.io import sha256


def load_data(features, reactions):
    with np.load(features) as f:
        metadata = json.loads(str(f["metadata"]))
        x = F.normalize(torch.tensor(f["embeddings"].astype("float32")), dim=-1)
    library = torch.load(reactions, map_location="cpu", weights_only=True)
    records = metadata["records"]
    assert len(records) == len(x) and torch.isfinite(x).all()
    ids = {rid: i for i, rid in enumerate(library["ids"])}
    group_names = {g: i for i, g in enumerate(sorted(set(library["groups"])))}
    groups = torch.tensor([group_names[g] for g in library["groups"]])
    gold = torch.tensor([ids[r["reaction_id"]] for r in records])
    for r in records:
        assert r["group_id"] == library["groups"][ids[r["reaction_id"]]], (
            "Reaction grouping changed"
        )
    split_groups = {
        s: {r["group_id"] for r in records if r["split"] == s}
        for s in ["train", "validation", "test", "diagnostic"]
    }
    keys = list(split_groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not split_groups[a] & split_groups[b], (
                f"Reaction groups overlap: {a}/{b}"
            )
    return dict(
        x=x,
        library=F.normalize(library["embeddings"].float(), dim=-1),
        gold=gold,
        groups=groups,
        records=records,
        encoder=metadata["encoder"],
        teacher=library["teacher"],
        provenance={
            "features_sha256": sha256(features),
            "reactions_sha256": sha256(reactions),
        },
    )


def metrics(ranks):
    ranks = np.asarray(ranks)
    return dict(
        queries=len(ranks),
        recall_at_1=float((ranks <= 1).mean()),
        recall_at_5=float((ranks <= 5).mean()),
        recall_at_10=float((ranks <= 10).mean()),
        mrr=float((1 / ranks).mean()),
        median_rank=float(np.median(ranks)),
    )


@torch.inference_mode()
def ranks(vectors, library, gold, groups, batch_size=128):
    result = []
    for start in range(0, len(vectors), batch_size):
        scores = vectors[start : start + batch_size] @ library.T
        relevant = groups[None, :] == groups[gold[start : start + len(scores)], None]
        target = scores.masked_fill(~relevant, -float("inf")).max(1).values
        result.extend((1 + (scores > target[:, None]).sum(1)).cpu().tolist())
    return np.array(result)


def device(name="auto"):
    if name != "auto":
        return name
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def interval(values, groups, repeats=2000):
    """Query-weighted percentile interval, resampling whole reaction groups."""
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(20260905)
    draw = rng.integers(len(sums), size=(repeats, len(sums)))
    estimates = sums[draw].sum(1) / counts[draw].sum(1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()
