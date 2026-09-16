"""Cache frozen text features. Encoder metadata travels with every artifact."""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data.io import read_jsonl, text_records, sha256
from settings import CACHE, PROCESSED, ENCODERS
from training.common import device


class TextEncoder:
    def __init__(self, config, requested_device="auto", offline=False):
        os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.device = device(requested_device)
        dtype = (
            torch.bfloat16
            if self.device == "cuda"
            else torch.float16
            if self.device == "mps"
            else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["revision"],
            padding_side="left" if config["pooling"] == "last" else "right",
            local_files_only=offline,
        )
        self.model = (
            AutoModel.from_pretrained(
                config["model"],
                revision=config["revision"],
                torch_dtype=dtype,
                attn_implementation="sdpa" if self.device == "cuda" else "eager",
                local_files_only=offline,
            )
            .eval()
            .to(self.device)
        )

    @torch.inference_mode()
    def __call__(self, texts):
        c = self.config
        t = self.tokenizer(
            [c["prefix"] + s for s in texts],
            padding=True,
            truncation=True,
            max_length=c["max_length"],
            return_tensors="pt",
        ).to(self.device)
        extra = {"use_cache": False} if "Qwen" in c["model"] else {}
        h = self.model(**t, **extra).last_hidden_state.float()
        if c["pooling"] == "last":
            v = h[:, -1]
        else:
            mask = t["attention_mask"][:, :, None]
            v = (h * mask).sum(1) / mask.sum(1)
        return F.normalize(v, dim=-1).cpu()


def embed(source, output, config, batch_size=24, requested_device="auto", reuse=None):
    records = text_records(read_jsonl(source))
    assert records, "No texts to embed"
    if output.exists():
        raise FileExistsError(
            f"Output exists: {output}; use a new filename and --reuse"
        )
    known = {}
    if reuse:
        with np.load(reuse) as f:
            m = json.loads(str(f["metadata"]))
            assert m["encoder"] == config, "Cannot reuse another encoder/pooling"
            known = {r["text"]: v for r, v in zip(m["records"], f["embeddings"])}
    needed = sorted({r["text"] for r in records} - set(known), key=len)
    start = time.monotonic()
    if needed:
        model = TextEncoder(config, requested_device)
        for i in range(0, len(needed), batch_size):
            texts = needed[i : i + batch_size]
            vectors = model(texts).numpy().astype("float16")
            known.update(zip(texts, vectors))
            print(
                f"Embedded {min(i + batch_size, len(needed))}/{len(needed)}", flush=True
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            embeddings=np.stack([known[r["text"]] for r in records]),
            metadata=json.dumps(
                dict(
                    encoder=config,
                    records=records,
                    source_sha256=sha256(source),
                    elapsed_seconds=time.monotonic() - start,
                )
            ),
        )
    tmp.replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=PROCESSED / "descriptions.jsonl")
    p.add_argument("--output", type=Path, default=CACHE / "text.npz")
    p.add_argument("--encoder", choices=list(ENCODERS), default="qwen4b")
    p.add_argument("--pooling", choices=["last", "mean"])
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--device", default="auto")
    p.add_argument("--reuse", type=Path)
    p.add_argument("--download-model", action="store_true")
    a = p.parse_args()
    config = {**ENCODERS[a.encoder]}
    if a.pooling:
        config["pooling"] = a.pooling
    if a.download_model:
        from huggingface_hub import snapshot_download

        snapshot_download(
            config["model"],
            revision=config["revision"],
            cache_dir=CACHE / "huggingface/hub",
        )
    else:
        embed(a.source, a.output, config, a.batch_size, a.device, a.reuse)
