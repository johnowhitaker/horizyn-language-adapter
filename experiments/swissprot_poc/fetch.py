"""Fetch cached enzyme candidates in batches, never one annotation call per protein."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import random
import time
import requests
import torch
from data.io import sha256
from settings import CACHE
from .common import ARTIFACTS, digest

FIELDS = "accession,protein_name,organism_id,organism_name,ec,cc_function,cc_catalytic_activity,cc_subcellular_location,sequence,sequence_version"


def fetch_batch(ids, directory):
    tag = digest("|".join(ids))[:20]
    path = directory / f"{tag}.tsv"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        assert sha256(path) == m["sha256"]
        return m
    query = "(reviewed:true) AND (ec:*) AND (cc_function:*) AND (" + " OR ".join("accession:" + p for p in ids) + ")"
    for attempt in range(4):
        try:
            r = requests.get("https://rest.uniprot.org/uniprotkb/search", params={
                "query": query, "format": "tsv", "fields": FIELDS, "size": 500,
            }, timeout=(15, 60))
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise ValueError(f"UniProt rejected batch: {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(r.text), delimiter="\t"))
            assert len(rows) == int(r.headers["x-total-results"]), "Unexpected pagination"
            assert {row["Entry"] for row in rows} <= set(ids)
            assert "Sequence" in r.text.splitlines()[0]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(r.text)
            tmp.replace(path)
            m = dict(requested=ids, returned=len(rows), release=r.headers["x-uniprot-release"],
                     url=r.url, sha256=sha256(path), retrieved_at=time.time(), path=path.name)
            meta.write_text(json.dumps(m, indent=2) + "\n")
            return m
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    assert 1 <= a.workers <= 4 and 1 <= a.batch_size <= 100
    pc = torch.load(CACHE / "proteins.pt", map_location="cpu", weights_only=True)
    ids = sorted(pc["ids"])
    random.Random(42).shuffle(ids)
    ids = ids[:a.candidates]
    directory = ARTIFACTS / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    batches = [ids[i:i+a.batch_size] for i in range(0, len(ids), a.batch_size)]
    metadata = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, m in enumerate(pool.map(lambda b: fetch_batch(b, directory), batches), 1):
            metadata.append(m)
            print(f"Batch {i}/{len(batches)}: {m['returned']} reviewed enzymes", flush=True)
    assert len({m["release"] for m in metadata}) == 1, "Mixed UniProt releases; use a fresh snapshot"
    manifest = dict(candidate_count=len(ids), batch_count=len(batches), release=metadata[0]["release"],
                    batches=metadata, protein_library_sha256=sha256(CACHE / "proteins.pt"))
    (ARTIFACTS / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
