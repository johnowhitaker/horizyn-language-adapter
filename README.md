# Horizyn language adapter

Describe a chemical reaction in words, then search for reactions and enzymes in
[Horizyn](https://github.com/dayhofflabs/horizyn)'s embedding space.

The idea is small: freeze a text encoder, embed several descriptions of each
reaction, and learn a projection to that reaction's existing Horizyn vector.
The same projected query can search both reaction and protein libraries.

```
reaction description → frozen text encoder → small adapter → Horizyn space
                              2,560 dims        512 dims      reactions / proteins
```

This repository keeps that pipeline explicit. There is no training framework,
experiment registry or frontend build system. Change the encoder, loss, data or
adapter directly and save each run in a new directory.

```
data/          download, split, generate descriptions, build teacher/text vectors
training/      adapter, training loop, evaluation, optional Modal wrapper
experiments/   small JSON configs; generated runs are ignored
web_demo/      FastAPI server and plain HTML/CSS/JavaScript
settings.py    paths and pinned encoder configurations
```

## Setup

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
cp .env.example .env        # fill in OPENROUTER_API_KEY for generation only
```

Data, weights, generated text, logs and `.env` are ignored by Git. A fresh clone
needs the preparation steps below; these artifacts are not bundled.

## Demo

```bash
uv run python -m web_demo.server
```

Open **http://127.0.0.1:8000/**. Stop with **Ctrl-C**; rerun the same command to
restart. One process serves the page and runs inference locally. No paid API
calls are made by the demo. Qwen 4B needs roughly 10 GB of working memory.

- Search with a description; reactions and proteins appear in two columns.
- Roll a diagnostic example to see its source reaction and rank.
- Switch to catalogue-assisted search to retrieve named equations directly.
- If the [Swiss-Prot experiment](experiments/swissprot_poc/README.md) is trained,
  choose **Protein descriptions · Swiss-Prot** to use its protein-description head.
- Compare several descriptions against a protein, with available AlphaFold views.

```bash
# Select another trained adapter or expose the running demo temporarily.
uv run python -m web_demo.server --checkpoint experiments/runs/my-run/adapter.pt
ngrok http 8000
```

The catalogue is optional and must use the same encoder as the adapter. Protein
names/structures use public UniProt and AlphaFold endpoints. 3Dmol.js is loaded
from a pinned CDN release; unavailable structures do not block search.

## Build the data

```bash
uv sync --extra teacher
uv run python -m data.download --proteins
uv run python -m data.prepare
uv run --extra teacher python -m data.targets --proteins
```

This downloads the pinned upstream Horizyn code, inference checkpoint and source
data into `data/raw/`, fetches Rhea names, groups equivalent reactions, and builds
the two frozen teacher libraries. It does not train Horizyn. Omit `--proteins`
from both commands for a reaction-only experiment. Teacher preparation is the
only stage that needs the optional upstream dependencies.

Generate a small sample, inspect it, then expand:

```bash
uv run python -m data.generate --limit 12 --workers 4 --budget-usd 1
uv run python -m data.generate --limit 3000 --workers 32 --budget-usd 8
```

The default generator is `z-ai/glm-5.3-flash` through OpenRouter. It returns five
styles: equation, precise, transformation, plain and informal. The accepted
prompt and tool schema are visible in `data/generate.py`. Review flags annotate
noisy examples; they do not exclude them.

`--budget-usd` is a **cumulative cap for that output directory**, not a per-run
allowance. Requests reserve their maximum cost before dispatch and settle to
reported usage afterward. Uncertain failures retain reservations. Completed
reaction IDs are skipped on resume. Use one generator process per directory.
To change the prompt/model, use a new `--out` directory with its own
`reactions.jsonl`; mixed generation signatures are rejected.

## Embed and train

```bash
uv run python -m data.embed --encoder qwen4b --output data/cache/text.npz
uv run python -m training.train --config experiments/baseline.json \
  --out experiments/runs/my-run
```

The default adapter is a centered linear map with cosine alignment, contrastive
retrieval loss and teacher-similarity distillation. `training/train.py` contains
the complete optimization loop. `experiments/train_negatives.json` is the
stricter contrastive variant using only training reactions as negatives.
`kind: "mlp"` enables a small two-layer head.

Qwen 4B, Qwen 0.6B and ModernBERT configurations live in `settings.py`. Encoder
identity, revision, pooling and prompt travel with the features and checkpoint.
Use `--pooling mean` for a pooling ablation, `--reuse previous.npz` to reuse
compatible text vectors, or edit the configuration to add another model. Choose
new output filenames for new datasets/encoders.

For GPU rental, the thin Modal wrapper calls the same functions:

```bash
uv sync --extra modal
uv run --extra modal modal run training/modal_run.py --task embed \
  --source data/processed/descriptions.jsonl --output data/cache/text.npz
uv run --extra modal modal run training/modal_run.py --task train \
  --config experiments/baseline.json --output experiments/runs/my-run
```

Modal uses one L4, no automatic retries and a one-hour timeout per call. It is
paid compute and requires your configured Modal account. After remote embedding,
download the encoder locally before running the demo:

```bash
uv run python -m data.embed --download-model
```

To enable catalogue-assisted search, embed canonical named equations separately:

```bash
uv run python -m data.embed --source data/processed/reactions.jsonl \
  --output data/cache/catalogue.npz
```

## Evaluation

```bash
uv run python -m training.evaluate \
  --checkpoint experiments/runs/my-run/adapter.pt \
  --split test --proteins --out experiments/runs/my-run/test.json
```

Training selects checkpoints on validation MRR only. Evaluation reports
reaction Recall@1/5/10, MRR, per-style and non-equation results, and a 95%
confidence interval that resamples whole reaction groups. `--proteins` adds
known-annotation protein retrieval. Per-query ranks are saved beside the report.

Keep test scores out of model selection. Reaction groups join exact structural
equivalents, reversals, matching equations and Rhea master IDs. They do **not**
enforce substrate-scaffold or enzyme-family novelty. The released inference
checkpoint already saw its source dataset: holdout here concerns the new
adapter. Targets average reaction directions, synthetic descriptions can be
wrong, and missing protein annotations are not evidence of inactivity.

Catalogue-assisted search is a different baseline: it indexes each candidate's
named equation, including held-out candidates. High catalogue scores therefore
do not demonstrate learned adapter generalization.

Keep examples explored interactively in a separate `diagnostic` split. Put demo
queries in `data/processed/demo_examples.jsonl` with `text` and `reaction_id`
fields; the server never samples the evaluation test set automatically.

## Credits

Built on [Horizyn](https://github.com/dayhofflabs/horizyn),
[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[ModernBERT Embed](https://huggingface.co/nomic-ai/modernbert-embed-base),
[Rhea](https://www.rhea-db.org/), UniProt and AlphaFold DB.
See [NOTICE.md](NOTICE.md) for upstream terms. The adapter code's license is
still to be chosen before publication.
