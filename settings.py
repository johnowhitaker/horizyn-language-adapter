"""One place for paths and the exact text-encoder contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
CACHE = ROOT / "data/cache"
RUNS = ROOT / "experiments/runs"
INSTRUCTION = (
    "Represent this biochemical reaction for retrieving matching reactions and enzymes."
)
ENCODERS = {
    "qwen4b": dict(
        model="Qwen/Qwen3-Embedding-4B",
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        pooling="last",
        prefix=f"Instruct: {INSTRUCTION}\nQuery: ",
        max_length=1024,
    ),
    "qwen06b": dict(
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        pooling="last",
        prefix=f"Instruct: {INSTRUCTION}\nQuery: ",
        max_length=1024,
    ),
    "modernbert": dict(
        model="nomic-ai/modernbert-embed-base",
        revision="d556a88e332558790b210f7bdbe87da2fa94a8d8",
        pooling="mean",
        prefix="search_query: ",
        max_length=1024,
    ),
}
