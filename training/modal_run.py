"""Thin optional GPU wrapper; the same Python functions run locally and remotely."""

from pathlib import Path
import json
import tempfile
import modal
from settings import ROOT, CACHE, PROCESSED, ENCODERS

app = modal.App("horizyn-language-adapter")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.56.2",
        "numpy==1.26.4",
        "requests>=2.32",
        "rdkit==2026.3.5",
    )
    .env({"HF_HOME": "/model-cache"})
    .add_local_file(ROOT / "settings.py", "/root/settings.py")
    .add_local_dir(
        ROOT / "data",
        "/root/data",
        ignore=["raw/**", "processed/**", "cache/**", "__pycache__/**"],
    )
    .add_local_dir(ROOT / "training", "/root/training", ignore=["__pycache__/**"])
)
cache = modal.Volume.from_name("horizyn-text-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/model-cache": cache},
)
def run(task, source, reactions, config):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = root / ("source.jsonl" if task == "embed" else "features.npz")
        src.write_bytes(source)
        if task == "embed":
            from data.embed import embed

            output = root / "text.npz"
            embed(src, output, config, 24, "cuda")
            cache.commit()
            return {"text.npz": output.read_bytes()}
        from training.train import train

        teacher = root / "reactions.pt"
        teacher.write_bytes(reactions)
        train(src, teacher, root / "run", config, "cuda")
        return {p.name: p.read_bytes() for p in (root / "run").iterdir()}


@app.local_entrypoint()
def main(
    task: str = "train",
    source: str = "",
    output: str = "",
    encoder: str = "qwen4b",
    config: str = "",
):
    if task not in ["train", "embed"]:
        raise ValueError("task must be train or embed")
    src = (
        Path(source)
        if source
        else (
            PROCESSED / "descriptions.jsonl" if task == "embed" else CACHE / "text.npz"
        )
    )
    if not output:
        raise ValueError("Choose --output (file for embed, run directory for train)")
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    cfg = (
        ENCODERS[encoder]
        if task == "embed"
        else json.loads(Path(config).read_text())
        if config
        else {}
    )
    result = run.remote(
        task,
        src.read_bytes(),
        (CACHE / "reactions.pt").read_bytes() if task == "train" else b"",
        cfg,
    )
    if task == "embed":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result["text.npz"])
    else:
        target.mkdir(parents=True)
        for name, blob in result.items():
            (target / name).write_bytes(blob)
    print("Saved", target)
