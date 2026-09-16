"""Build frozen teacher libraries using the pinned upstream Horizyn implementation."""

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from data.io import read_jsonl
from settings import RAW, PROCESSED, CACHE
from training.common import device


@torch.inference_mode()
def build(
    upstream, checkpoint, output, requested_device="auto", proteins=False, limit=0
):
    # Keep the upstream package outside this repository's source; do not reimplement its chemistry.
    sys.path.insert(0, str(upstream.resolve()))
    from horizyn.config import load_config
    from horizyn.lightning_module import HorizynLitModule
    from horizyn.datasets.hdf5 import EmbedDataset

    spec = importlib.util.spec_from_file_location(
        "horizyn_predict", upstream / "scripts/predict.py"
    )
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    config = load_config(str(upstream / "configs/sota.yaml"))
    dev = device(requested_device)
    torch.set_num_threads(4)
    model = (
        HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
        .eval()
        .to(dev)
    )
    with checkpoint.open("rb") as f:
        checksum = hashlib.file_digest(f, "md5").hexdigest()
    teacher = dict(checkpoint_md5=checksum, bidirectional=True)
    meta = {r["reaction_id"]: r for r in read_jsonl(PROCESSED / "reactions.jsonl")}
    reactions = {}
    for split in ["train", "test"]:
        with (RAW / f"{split}_rxns.csv").open() as f:
            reactions.update({r["reaction_id"]: r for r in csv.DictReader(f)})
    rows = sorted(reactions.values(), key=lambda r: r["reaction_id"])
    if limit:
        rows = rows[:limit]
    if (output / "reactions.pt").exists():
        raise FileExistsError("Reaction library exists; choose --out for a new library")
    output.mkdir(parents=True, exist_ok=True)
    vectors = []
    for i, row in enumerate(rows):
        fingerprints, _ = predict.build_reaction_fingerprint(
            row["reaction_smiles"], config, bidirectional=True
        )
        vectors.append(model.model.query_encoder(fingerprints.to(dev)).mean(0).cpu())
        if i % 500 == 0:
            print("Reactions", i, "/", len(rows), flush=True)
    metadata = [
        meta.get(
            r["reaction_id"],
            {**r, "group_id": r["reaction_id"], "equation": r["reaction_id"], "ec": ""},
        )
        for r in rows
    ]
    torch.save(
        dict(
            ids=[r["reaction_id"] for r in rows],
            embeddings=F.normalize(torch.stack(vectors), dim=-1),
            metadata=metadata,
            groups=[r["group_id"] for r in metadata],
            teacher=teacher,
        ),
        output / "reactions.pt",
    )
    if proteins:
        if (output / "proteins.pt").exists():
            raise FileExistsError("Protein library exists")
        source = EmbedDataset(str(RAW / "prots_t5.h5"), in_memory=True)
        ids = source.keys[:limit] if limit else source.keys
        vectors = []
        for start in range(0, len(ids), 2048):
            vectors.append(
                model.model.target_encoder(
                    source.data[start : min(start + 2048, len(ids))].to(dev)
                ).cpu()
            )
        torch.save(
            dict(
                ids=ids,
                embeddings=F.normalize(torch.cat(vectors), dim=-1),
                teacher=teacher,
            ),
            output / "proteins.pt",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, default=RAW / "horizyn")
    p.add_argument("--checkpoint", type=Path, default=RAW / "horizyn_v1_0_inf.ckpt")
    p.add_argument("--out", type=Path, default=CACHE)
    p.add_argument("--device", default="auto")
    p.add_argument("--proteins", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    build(a.upstream, a.checkpoint, a.out, a.device, a.proteins, a.limit)
