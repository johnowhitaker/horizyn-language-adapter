"""Train small protein adapters and compare held-out retrieval with the reaction adapter."""
import argparse
from collections import defaultdict
import csv
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from data.io import read_jsonl, sha256
from settings import CACHE, RAW, RUNS
from training.model import Adapter, load_adapter
from training.common import metrics, interval, ranks as reaction_ranks
from .common import ARTIFACTS


def load_features(path):
    with np.load(path) as a:
        return F.normalize(torch.from_numpy(a["embeddings"].astype("float32")), dim=-1), json.loads(str(a["metadata"]))


def load_dataset():
    base = ARTIFACTS / "dataset"
    manifest = json.loads((base / "manifest.json").read_text())
    assert sha256(base / "texts.jsonl") == manifest["texts_sha256"]
    assert sha256(base / "targets.pt") == manifest["targets_sha256"]
    assert sha256(base / "proteins.jsonl") == manifest["proteins_sha256"]
    x, meta = load_features(ARTIFACTS / "features/protein.npz")
    assert meta["source_sha256"] == manifest["texts_sha256"]
    target = torch.load(base / "targets.pt", map_location="cpu", weights_only=True)
    proteins = read_jsonl(base / "proteins.jsonl")
    records = meta["records"]
    lookup = {p: i for i, p in enumerate(target["ids"])}
    gold = torch.tensor([lookup[r["protein_id"]] for r in records])
    tg = {g: i for i, g in enumerate(sorted(set(target["text_groups"])))}
    text_groups = torch.tensor([tg[g] for g in target["text_groups"]])
    split_groups, split_texts = defaultdict(set), defaultdict(set)
    for i, r in enumerate(records):
        assert r["group_id"] == target["groups"][gold[i]]
        assert r["text_group"] == target["text_groups"][gold[i]]
        split_groups[r["split"]].add(r["group_id"])
        split_texts[r["split"]].add(r["text_group"])
    for a in split_groups:
        for b in split_groups:
            if a != b:
                assert split_groups[a].isdisjoint(split_groups[b])
                assert split_texts[a].isdisjoint(split_texts[b])
    assert len(x) == len(records) and torch.isfinite(x).all()
    return dict(x=x, meta=meta, records=records, gold=gold, text_groups=text_groups,
        target=target, library=F.normalize(target["embeddings"].float(), dim=-1), proteins=proteins, manifest=manifest)


@torch.inference_mode()
def retrieval_ranks(z, library, positive_sets, batch_size=32):
    """Best relevant result; tied non-relevant candidates get their average position."""
    out = []
    for start in range(0, len(z), batch_size):
        scores = (z[start:start + batch_size] @ library.T).cpu()
        for row, positives in zip(scores, positive_sets[start:start + batch_size]):
            assert positives
            positives = sorted(set(positives))
            best = row[positives].max()
            tied = (row - best).abs() <= 1e-7
            tied[positives] = False
            out.append(1 + int((row > best + 1e-7).sum()) + .5 * int(tied.sum()))
    return np.array(out)


def text_positives(data, ids):
    groups = defaultdict(list)
    for i, g in enumerate(data["target"]["text_groups"]):
        groups[g].append(i)
    return [groups[data["records"][i]["text_group"]] for i in ids]


def save_checkpoint(model, data, output, config, validation, epoch):
    checkpoint = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        model_config=model.config, encoder=data["meta"]["encoder"], teacher=data["target"]["teacher"],
        config=config, validation=validation, best_epoch=epoch, target_type="protein",
        provenance=dict(dataset=data["manifest"], features_sha256=sha256(ARTIFACTS / "features/protein.npz")))
    torch.save(checkpoint, output / "adapter.pt")


def train(requested_device="mps"):
    torch.set_num_threads(6)
    torch.manual_seed(42)
    data = load_dataset()
    output = ARTIFACTS / "runs"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    tr = [i for i, r in enumerate(data["records"]) if r["split"] == "train"]
    va = [i for i, r in enumerate(data["records"]) if r["split"] == "validation"]
    assert tr and va
    # All negative and distillation targets are from the training split.
    candidates = torch.unique(data["gold"][tr])
    val_positives = text_positives(data, va)
    summaries = []
    center = data["x"][tr].mean(0)
    started = time.monotonic()
    # Fixed small ridge baseline grid; both use training rows only.
    rx = F.normalize(data["x"][tr] - center, dim=-1)
    rx = torch.cat([rx, torch.ones(len(rx), 1)], dim=1)
    ry = data["library"][data["gold"][tr]]
    gram, rhs = rx.T @ rx, rx.T @ ry
    for regularization in [.1, 1.0]:
        name = f"ridge-{regularization:g}"
        run = output / name
        run.mkdir()
        penalty = torch.eye(gram.shape[0]) * regularization
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(gram + penalty, rhs)
        model = Adapter(data["x"].shape[1])
        with torch.no_grad():
            model.center.copy_(center)
            model.net.weight.copy_(weights[:-1].T)
            model.net.bias.copy_(weights[-1])
        result = metrics(retrieval_ranks(model(data["x"][va]), data["library"], val_positives))
        save_checkpoint(model, data, run, dict(kind="ridge", regularization=regularization), result, 0)
        summaries.append(dict(name=name, validation=result))
        print(json.dumps(summaries[-1]), flush=True)
    model = Adapter(data["x"].shape[1]).to(requested_device)
    model.center.copy_(center.to(requested_device))
    x, library = data["x"].to(requested_device), data["library"].to(requested_device)
    gold, text_groups = data["gold"].to(requested_device), data["text_groups"].to(requested_device)
    candidates = candidates.to(requested_device)
    tr_tensor = torch.tensor(tr, device=requested_device)
    config = dict(kind="linear", cosine=1., contrast=.5, distill=1., temperature=.07,
                  teacher_temperature=.05, epochs=64, patience=6, lr=.002, batch_size=256,
                  negatives="train", seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    run = output / "linear"
    run.mkdir()
    history, best, stale = [], -1., 0
    for epoch in range(config["epochs"]):
        model.train()
        for ids in tr_tensor[torch.randperm(len(tr_tensor), device=requested_device)].split(config["batch_size"]):
            z = model(x[ids])
            targets = library[gold[ids]]
            scores = z @ library[candidates].T
            logits = scores / config["temperature"]
            positive = text_groups[gold[ids], None] == text_groups[candidates][None, :]
            contrast = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positive, -float("inf")), 1)).mean()
            t = config["teacher_temperature"]
            teacher = (targets @ library[candidates].T / t).softmax(1)
            loss = (1 - (z * targets).sum(-1)).mean() + .5 * contrast + F.kl_div(
                (scores / t).log_softmax(1), teacher, reduction="batchmean")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        if epoch == 0 or (epoch + 1) % 4 == 0:
            model.eval()
            result = metrics(retrieval_ranks(model(x[va]), library, val_positives))
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach()), **result))
            print(json.dumps(history[-1]), flush=True)
            if result["mrr"] > best:
                best, stale = result["mrr"], 0
                save_checkpoint(model, data, run, config, result, epoch + 1)
            else:
                stale += 1
            if stale >= config["patience"]:
                break
    (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    cp = torch.load(run / "adapter.pt", map_location="cpu", weights_only=True)
    summaries.append(dict(name="linear", validation=cp["validation"], best_epoch=cp["best_epoch"]))
    winner = max(summaries, key=lambda r: r["validation"]["mrr"])
    selected = ARTIFACTS / "selected"
    selected.mkdir()
    (selected / "adapter.pt").write_bytes((output / winner["name"] / "adapter.pt").read_bytes())
    report = dict(selection_metric="validation function-text-group MRR over the selected protein cohort",
        selected=winner["name"], runs=summaries, elapsed_seconds=time.monotonic() - started,
        training_proteins=len(candidates), training_pairs=len(tr), validation_queries=len(va))
    (ARTIFACTS / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


@torch.inference_mode()
def evaluate(split="test"):
    out = ARTIFACTS / f"{split}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(6)
    data = load_dataset()
    ids = [i for i, r in enumerate(data["records"]) if r["split"] == split]
    assert ids
    rx, rm = load_features(ARTIFACTS / "features/reaction.npz")
    reaction, rcp = load_adapter(RUNS / "best/adapter.pt")
    protein, pcp = load_adapter(ARTIFACTS / "selected/adapter.pt")
    assert rm["encoder"] == rcp["encoder"] and pcp["encoder"] == data["meta"]["encoder"]
    assert pcp["provenance"]["dataset"] == data["manifest"]
    assert rm["source_sha256"] == data["manifest"]["texts_sha256"]
    lookup = {(r["protein_id"], r["style"]): i for i, r in enumerate(rm["records"])}
    ri = [lookup[(data["records"][i]["protein_id"], data["records"][i]["style"])] for i in ids]
    assert all(rm["records"][j]["text"] == data["records"][i]["text"] for i, j in zip(ids, ri))
    vectors = {"protein_adapter": protein(data["x"][ids]), "reaction_adapter": reaction(rx[ri])}
    pc = torch.load(CACHE / "proteins.pt", map_location="cpu", weights_only=True)
    assert pcp["teacher"] == rcp["teacher"] == pc["teacher"]
    assert sha256(CACHE / "proteins.pt") == data["manifest"]["protein_library_sha256"]
    full = F.normalize(pc["embeddings"].float(), dim=-1)
    full_lookup = {p: i for i, p in enumerate(pc["ids"])}
    pids = [data["records"][i]["protein_id"] for i in ids]
    groups = np.array([data["records"][i]["group_id"] for i in ids])
    exact = [[int(data["gold"][i])] for i in ids]
    full_exact = [[full_lookup[p]] for p in pids]
    reaction_proteins, protein_reactions = defaultdict(set), defaultdict(set)
    for name in ["train_pairs.csv", "test_pairs.csv"]:
        with (RAW / name).open() as f:
            for row in csv.DictReader(f):
                if row["protein_id"] in full_lookup:
                    reaction_proteins[row["reaction_id"]].add(full_lookup[row["protein_id"]])
                    protein_reactions[row["protein_id"]].add(row["reaction_id"])
    known = [sorted(set.union({full_lookup[p]}, *(reaction_proteins[r] for r in protein_reactions[p]))) for p in pids]
    report = dict(split=split, queries=len(ids), groups=len(set(groups)), cohort_candidates=len(data["library"]),
        unique_descriptions=len({data["records"][i]["text"] for i in ids}),
        full_candidates=len(full), selected=json.loads((ARTIFACTS / "selection.json").read_text())["selected"],
        protein_checkpoint_sha256=sha256(ARTIFACTS / "selected/adapter.pt"),
        reaction_checkpoint_sha256=sha256(RUNS / "best/adapter.pt"), models={})
    rank_arrays = {}
    function_groups = np.array([data["records"][i]["text_group"] for i in ids])
    _, function_inverse = np.unique(function_groups, return_inverse=True)
    function_counts = np.bincount(function_inverse)
    for name, z in vectors.items():
        results = {}
        for metric, lib, positives in [
            ("cohort_exact", data["library"], exact),
            ("cohort_same_function_text", data["library"], text_positives(data, ids)),
            ("full_exact", full, full_exact),
            ("full_shared_known_reaction", full, known),
        ]:
            ranks = retrieval_ranks(z, lib, positives)
            rank_arrays[f"{name}_{metric}"] = ranks
            results[metric] = {**metrics(ranks), "recall_at_10_95ci": interval((ranks <= 10).astype(float), groups)}
            results[metric]["function_balanced_recall_at_10"] = float((
                np.bincount(function_inverse, weights=(ranks <= 10).astype(float)) / function_counts).mean())
            print(name, metric, json.dumps(results[metric]), flush=True)
        report["models"][name] = results
    report["paired_delta_full_exact_recall_at_10_95ci"] = interval(
        (rank_arrays["protein_adapter_full_exact"] <= 10).astype(float) -
        (rank_arrays["reaction_adapter_full_exact"] <= 10).astype(float), groups)
    # Transfer check on the same protein-description queries using known protein/reaction labels.
    rc = torch.load(CACHE / "reactions.pt", map_location="cpu", weights_only=True)
    rlookup = {r: i for i, r in enumerate(rc["ids"])}
    relevant = []
    for p in pids:
        rg = {rc["groups"][rlookup[r]] for r in protein_reactions[p] if r in rlookup}
        relevant.append([i for i, g in enumerate(rc["groups"]) if g in rg])
    usable = [i for i, pos in enumerate(relevant) if pos]
    report["reaction_transfer_queries"] = len(usable)
    for name, z in vectors.items():
        ranks = retrieval_ranks(z[usable], F.normalize(rc["embeddings"].float(), dim=-1), [relevant[i] for i in usable])
        report["models"][name]["reaction_transfer"] = metrics(ranks)
    if split == "test":
        # Reverse transfer: keep the original reaction test descriptions fixed.
        source_meta = json.loads((ARTIFACTS / "features/reaction_source.json").read_text())
        assert sha256(CACHE / "text.npz") == source_meta["sha256"]
        original_x, original_meta = load_features(CACHE / "text.npz")
        qx, qm = load_features(ARTIFACTS / "features/reaction_queries.npz")
        original_ids = [i for i, r in enumerate(original_meta["records"]) if r["split"] == "test"]
        assert qm["records"] == [original_meta["records"][i] for i in original_ids]
        assert qm["encoder"] == pcp["encoder"] and original_meta["encoder"] == rcp["encoder"]
        rg_names = {g: i for i, g in enumerate(sorted(set(rc["groups"])))}
        rg = torch.tensor([rg_names[g] for g in rc["groups"]])
        gold = torch.tensor([rlookup[r["reaction_id"]] for r in qm["records"]])
        report["original_reaction_test"] = {}
        for name, z in [("protein_adapter", protein(qx)), ("reaction_adapter", reaction(original_x[original_ids]))]:
            ranks = reaction_ranks(z, F.normalize(rc["embeddings"].float(), dim=-1), gold, rg)
            report["original_reaction_test"][name] = metrics(ranks)
    report["limitations"] = [
        "Held out from this protein adapter by approximate MMseqs sequence clusters and identical function text; this does not guarantee absence of every remote homolog. Pretrained models may have seen these proteins.",
        "The existing reaction adapter has different training data and is not a matched-data control.",
        "Shared-known-reaction relevance accepts any annotated reaction shared with the source protein; labels are incomplete and functions may be broader.",
        "Exact-protein retrieval is strict: descriptions often cannot distinguish functional orthologs. Ties use average non-relevant position.",
        "Curated function paragraphs are not an independent benchmark of casual natural-language queries.",
    ]
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(ARTIFACTS / f"{split}_ranks.npz", protein_ids=pids, groups=groups, **rank_arrays)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["train", "evaluate"])
    p.add_argument("--device", default="mps")
    p.add_argument("--split", choices=["test", "diagnostic", "validation"], default="test")
    a = p.parse_args()
    train(a.device) if a.task == "train" else evaluate(a.split)
