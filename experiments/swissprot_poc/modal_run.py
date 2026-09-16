"""Bounded L4 jobs for frozen text features and adapter fitting, without retries."""
import json
import tempfile
import time
from pathlib import Path
import modal
import numpy as np
from settings import ROOT as PROJECT, CACHE
from data.io import sha256
from .common import ROOT, ARTIFACTS

app = modal.App("horizyn-swissprot-poc")
cache = modal.Volume.from_name("horizyn-text-model-cache", create_if_missing=True)


def ignore_non_source(path):
    return path.suffix != ".py" or any(part in {
        "raw", "processed", "cache", "artifacts", "tools", "__pycache__"
    } for part in path.parts)


image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.56.2", "numpy==1.26.4", "requests>=2.32,<3")
    .env({"HF_HOME": "/model-cache"})
    .add_local_file(PROJECT / "settings.py", "/root/settings.py")
    .add_local_dir(PROJECT / "data", "/root/data", ignore=ignore_non_source)
    .add_local_dir(PROJECT / "training", "/root/training", ignore=ignore_non_source)
    .add_local_dir(ROOT, "/root/experiments/swissprot_poc", ignore=ignore_non_source))


@app.function(image=image, gpu="L4", cpu=2, memory=16384, timeout=3600,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/model-cache": cache})
def run(source, reaction_queries):
    from experiments.swissprot_poc.embed import embed
    import torch
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = root / "texts.jsonl"
        src.write_bytes(source)
        embed(src, root / "features", "cuda", reaction_queries)
        cache.commit()
        (root / "features/execution.json").write_text(json.dumps(dict(
            gpu=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic() - start,
            timeout_seconds=3600, automatic_retries=0)))
        return {p.name: p.read_bytes() for p in (root / "features").iterdir()}


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2)
def train_run(files):
    from experiments.swissprot_poc import learning
    import torch
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, blob in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        learning.ARTIFACTS = root
        start = time.monotonic()
        learning.train("cuda")
        result = {str(p.relative_to(root)): p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file()}
        result["selected/adapter.pt"] = (root / "selected/adapter.pt").read_bytes()
        result["selection.json"] = (root / "selection.json").read_bytes()
        result["training_execution.json"] = json.dumps(dict(
            gpu=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic() - start,
            timeout_seconds=1800, automatic_retries=0)).encode()
        return result


@app.local_entrypoint()
def main(task: str = "embed"):
    if task == "train":
        if (ARTIFACTS / "runs").exists() or (ARTIFACTS / "selected").exists():
            raise FileExistsError("Training outputs already exist")
        names = ["dataset/manifest.json", "dataset/texts.jsonl", "dataset/proteins.jsonl",
                 "dataset/targets.pt", "features/protein.npz"]
        result = train_run.remote({name: (ARTIFACTS / name).read_bytes() for name in names})
        for name, blob in result.items():
            path = ARTIFACTS / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        print(f"Saved training results to {ARTIFACTS}")
        return
    if task != "embed":
        raise ValueError("task must be embed or train")
    output = ARTIFACTS / "features"
    if output.exists():
        raise FileExistsError(output)
    with np.load(CACHE / "text.npz") as f:
        records = json.loads(str(f["metadata"]))["records"]
    reaction_queries = [r for r in records if r["split"] == "test"]
    result = run.remote((ARTIFACTS / "dataset/texts.jsonl").read_bytes(), reaction_queries)
    output.mkdir(parents=True)
    for name, blob in result.items():
        (output / name).write_bytes(blob)
    (output / "reaction_source.json").write_text(json.dumps({"sha256": sha256(CACHE / "text.npz")}))
    print(f"Saved features to {output}")
