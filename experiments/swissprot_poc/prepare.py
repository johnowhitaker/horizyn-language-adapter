"""Match reaction training counts with sequence-verified Swiss-Prot enzyme descriptions."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import random
import re
import subprocess
import numpy as np
import torch
from data.io import read_jsonl, write_jsonl, sha256
from settings import CACHE
from .common import ARTIFACTS, ROOT, clean_function, digest, fasta_records, text_key


def components(rows, sequence_pairs):
    parent = {r["protein_id"]: r["protein_id"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for a, b in sequence_pairs:
        union(a, b)
    seen = {}
    for r in rows:
        for key in [("text", text_key(r["function"])), ("sequence", r["sequence_sha256"])]:
            if key in seen:
                union(r["protein_id"], seen[key])
            seen[key] = r["protein_id"]
    return {p: find(p) for p in parent}


def split_for(group):
    x = int(digest("swissprot-poc-42:" + group)[:8], 16) / 2**32
    return "train" if x < .78 else "validation" if x < .87 else "test" if x < .96 else "diagnostic"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-fasta", type=str, required=True)
    p.add_argument("--match-features", default=str(CACHE / "text.npz"))
    p.add_argument("--mmseqs", default=str(ROOT / "tools/mmseqs/bin/mmseqs"))
    p.add_argument("--resume-clustering", action="store_true",
                   help="Reuse an interrupted clustering pass on exactly the same input")
    a = p.parse_args()
    output = ARTIFACTS / "dataset"
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared dataset exists; preserve it or choose a new experiment")
    manifest = json.loads((ARTIFACTS / "download.json").read_text())
    with np.load(a.match_features) as features:
        reference = json.loads(str(features["metadata"]))
    requested = Counter(r["split"] for r in reference["records"])
    # Diagnostics are for the demo, not fitting or model selection.
    requested["diagnostic"] = 1000
    pc = torch.load(CACHE / "proteins.pt", map_location="cpu", weights_only=True)
    assert sha256(CACHE / "proteins.pt") == manifest["protein_library_sha256"]
    refs = dict(fasta_records(a.reference_fasta))
    rejected = Counter()
    candidates = {}
    for batch in manifest["batches"]:
        path = ARTIFACTS / "raw" / batch["path"]
        assert sha256(path) == batch["sha256"]
        with path.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                pid, seq = r["Entry"], r["Sequence"]
                if refs.get(pid) != seq:
                    rejected["sequence_mismatch_or_missing"] += 1
                    continue
                text = clean_function(r["Function [CC]"])
                if not 50 <= len(seq) <= 1500 or not 15 <= len(text.split()) <= 250:
                    rejected["length_filter"] += 1
                    continue
                if "[Cleaved into:" in r["Protein names"]:
                    rejected["polyprotein"] += 1
                    continue
                ecs = sorted(set(x.strip() for x in r["EC number"].split(";") if x.strip()))
                if not ecs or "uncharacterized" in r["Protein names"].lower():
                    rejected["uninformative"] += 1
                    continue
                candidates[pid] = dict(protein_id=pid, name=r["Protein names"].split(" (")[0],
                    organism=r["Organism"], taxon=r["Organism (ID)"], ec=ecs,
                    function=text, function_evidence=r["Function [CC]"],
                    catalytic_activity=r["Catalytic activity"], sequence=seq,
                    sequence_sha256=digest(seq), sequence_version=r["Sequence version"],
                    experimental_function="ECO:0000269" in r["Function [CC]"])
    # Order across EC numbers before deterministic subsampling within each split.
    # One protein contributes one real description; no generated paraphrases.
    buckets = defaultdict(list)
    for r in sorted(candidates.values(), key=lambda r: r["protein_id"]):
        buckets[r["ec"][0]].append(r)
    rng = random.Random(42)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    eligible, seqs = [], set()
    while any(buckets.values()):
        for key in keys:
            if not buckets[key]:
                continue
            r = buckets[key].pop()
            if r["sequence_sha256"] in seqs:
                continue
            eligible.append(r)
            seqs.add(r["sequence_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    fasta = output / "eligible.fasta"
    fasta.write_text("".join(f">{r['protein_id']}\n{r['sequence']}\n" for r in eligible))
    resume = output / "clustering_resume.json"
    fingerprint = dict(input_sha256=sha256(fasta))
    if a.resume_clustering:
        assert json.loads(resume.read_text()) == fingerprint, "Clustering input changed"
    resume.write_text(json.dumps(fingerprint))
    command = [a.mmseqs, "easy-linclust", str(fasta), str(output / "clusters"), str(output / "linclust_tmp"),
               "--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "0", "--cluster-mode", "1",
               "--alignment-mode", "3",
               "--max-rejected", "128",
               "--kmer-per-seq", "80",
               "--linclust-version", "1",
               "--threads", "6", "--split-memory-limit", "4G", "-v", "3"]
    if a.resume_clustering:
        command.extend(["--force-reuse", "1"])
    print(f"Clustering {len(eligible)} eligible proteins", flush=True)
    with (output / "clustering.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    pairs = [line.split() for line in (output / "clusters_cluster.tsv").read_text().splitlines()]
    assert {b for a, b in pairs} == {r["protein_id"] for r in eligible}, "Incomplete clustering output"
    group = components(eligible, pairs)
    selected, counts = [], Counter()
    for r in eligible:
        split = split_for(group[r["protein_id"]])
        if counts[split] < requested[split]:
            selected.append(r)
            counts[split] += 1
    if any(counts[s] < requested[s] for s in requested):
        raise ValueError(f"Not enough eligible proteins per split: {dict(counts)} vs {dict(requested)}; extend the candidate download")
    selected.sort(key=lambda r: r["protein_id"])
    (output / "proteins.fasta").write_text("".join(f">{r['protein_id']}\n{r['sequence']}\n" for r in selected))
    records = []
    for r in selected:
        r["group_id"] = group[r["protein_id"]]
        r["split"] = split_for(r["group_id"])
        r["text_group"] = digest(text_key(r["function"]))
        base = {k: r[k] for k in ["protein_id", "group_id", "split", "text_group"]}
        records.append({**base, "style": "name_function", "text": r["name"] + ". " + r["function"]})
    write_jsonl(output / "proteins.jsonl", selected)
    write_jsonl(output / "texts.jsonl", records)
    write_jsonl(output / "demo_examples.jsonl", [dict(text=r["name"] + ". " + r["function"], protein_id=r["protein_id"],
        name=r["name"], organism=r["organism"], ec="; ".join(r["ec"])) for r in selected if r["split"] == "diagnostic"])
    lookup = {pid: i for i, pid in enumerate(pc["ids"])}
    torch.save(dict(ids=[r["protein_id"] for r in selected],
        embeddings=pc["embeddings"][[lookup[r["protein_id"]] for r in selected]],
        teacher=pc["teacher"], groups=[r["group_id"] for r in selected],
        text_groups=[r["text_group"] for r in selected]), output / "targets.pt")
    report = dict(release=manifest["release"], candidate_requests=manifest["candidate_count"],
        eligible=len(candidates), rejected=dict(rejected), proteins=len(selected), text_pairs=len(records),
        unique_functions=len({text_key(r["function"]) for r in selected}),
        unique_texts_by_split={s: len({r["text"] for r in records if r["split"] == s}) for s in counts},
        matched_sample_counts=dict(requested), matched_features_sha256=sha256(a.match_features),
        ec_numbers=len({e for r in selected for e in r["ec"]}),
        eligible_sequence_clusters=len({a for a, b in pairs}),
        split_groups_by_split={s: len({r["group_id"] for r in selected if r["split"] == s}) for s in counts},
        splits=dict(Counter(r["split"] for r in selected)),
        experimental_function=sum(r["experimental_function"] for r in selected),
        reference_fasta_sha256=sha256(a.reference_fasta), protein_library_sha256=manifest["protein_library_sha256"],
        mmseqs_version=subprocess.check_output([a.mmseqs, "version"], text=True).strip(),
        clustering_input_sha256=sha256(fasta),
        clustering_command=command, texts_sha256=sha256(output / "texts.jsonl"),
        proteins_sha256=sha256(output / "proteins.jsonl"),
        targets_sha256=sha256(output / "targets.pt"))
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
