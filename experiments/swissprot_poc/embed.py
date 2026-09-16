"""Embed both query instructions with one frozen Qwen instance."""
import json
import time
import numpy as np
from data.embed import TextEncoder
from data.io import sha256
from settings import ENCODERS
from .common import ENCODER


def embed(source, output, requested_device="auto", reaction_queries=None):
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = TextEncoder(ENCODER, requested_device)
    jobs = [
        ("protein", ENCODER, rows),
        ("reaction", ENCODERS["qwen4b"], [r for r in rows if r["split"] != "train"]),
    ]
    if reaction_queries is not None:
        jobs.append(("reaction_queries", ENCODER, reaction_queries))
    for name, config, records in jobs:
        start = time.monotonic()
        texts = sorted({r["text"] for r in records}, key=lambda t: (len(t), t))
        known = {}
        truncations = 0
        for offset in range(0, len(texts), 24):
            batch = texts[offset:offset + 24]
            lengths = model.tokenizer([config["prefix"] + t for t in batch], truncation=False)["input_ids"]
            truncations += sum(len(t) > config["max_length"] for t in lengths)
            known.update(zip(batch, model(batch, prefix=config["prefix"]).numpy().astype("float16")))
            if offset % 504 == 0:
                print(f"{name}: {min(offset + len(batch), len(texts))}/{len(texts)}, {time.monotonic() - start:.0f}s", flush=True)
        np.savez_compressed(output / f"{name}.npz", embeddings=np.stack([known[r["text"]] for r in records]),
            metadata=json.dumps(dict(encoder=config, records=records,
                source_sha256=sha256(source) if name != "reaction_queries" else None,
                unique_texts=len(texts), truncated_texts=truncations, elapsed_seconds=time.monotonic() - start)))
        print(f"{name}: finished {len(records)} rows, {truncations} truncated", flush=True)
