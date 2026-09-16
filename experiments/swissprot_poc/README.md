# Swiss-Prot protein-description adapter

This experiment maps existing UniProtKB/Swiss-Prot enzyme descriptions into
Horizyn's frozen protein embedding space. It uses the same pinned
Qwen3-Embedding-4B encoder and 2,560 → 512 adapter architecture as the reaction
experiment. The encoder is frozen; only the small adapter is trained.

The requested scale is **46,845 training pairs**, matching the existing reaction
text cache exactly, with **4,135 validation** and **4,345 test** pairs. Each
Swiss-Prot pair uses one distinct protein sequence and its protein name plus
function paragraph. A further 1,000 proteins are reserved for interactive demo
examples. These are existing annotations downloaded in batches, not annotation
generation requests or synthetic paraphrases.

## Results — 2026-09-15

The selected linear adapter is from epoch 24; early stopping ended fitting at
epoch 48. Its validation MRR was 0.260, versus 0.101 and 0.077 for the ridge
baselines. The formal Swiss-Prot test contains 4,345 protein queries, 1,081
unique descriptions and 656 split components.

All values below are Recall@10. Both models receive the same queries within
each row and search the same library.

| Test | Reaction adapter | Swiss-Prot adapter |
| --- | ---: | ---: |
| Swiss-Prot text → exact source protein, full 216,132-protein library | 3.15% | **7.50%** |
| Swiss-Prot text → any protein sharing a known reaction, full library | 27.57% | **62.00%** |
| Swiss-Prot text → known reaction, 11,797-reaction library | 53.26% | **60.24%** |
| Original reaction text → target reaction, 11,797-reaction library | **58.64%** | 10.08% |

For exact-protein retrieval, median rank improves from 1,195 to 248; the paired
95% component-bootstrap interval for the Recall@10 gain is **+2.75 to +6.30
percentage points**. Function-balanced Recall@10 for shared-known-reaction
protein retrieval is 19.13% versus 62.95%. The shared-reaction metric is lenient:
matching any known reaction is sufficient, and six Swiss-Prot test proteins
lack a usable reaction-transfer label.

This supports keeping separate search options: the new head is substantially
better on curated protein descriptions, while the original head is much better
on the original reaction descriptions. It is not evidence that the new head
universally improves informal queries.

| Training-set property | Reaction experiment | Swiss-Prot experiment |
| --- | ---: | ---: |
| Training pairs | 46,845 | 46,845 |
| Unique input descriptions | 46,843 | 10,654 |
| Distinct training targets | 9,397 reactions | 46,845 proteins |

The released dataset has 56,325 distinct sequences, 11,111 distinct cleaned
function paragraphs and 3,742 EC numbers. It uses UniProt release `2026_03`;
9,103 selected entries have experimental function evidence, and the rest are
not assumed to be experimentally established. No input descriptions were
truncated. Qwen embedding took 736 seconds on an L4; the completed GPU fitting
job took 218 seconds. An earlier local fitting attempt is preserved separately
and was not included in final checkpoint selection.

Machine-readable results and per-query ranks are in `artifacts/test.json` and
`artifacts/test_ranks.npz`; the selection record is `artifacts/selection.json`.
The winning checkpoint is `artifacts/selected/adapter.pt` (SHA-256
`63624367c622bc9d01605e90602d924115bedc17312644dcbfdc6b411e702361`).

Live smoke checks of informal peroxide breakdown, photosynthesis carbon fixation,
and lipid hydrolysis queries retrieved catalases, Rubisco and lipases. These
three examples are qualitative checks, not a further benchmark; responses are
saved in `artifacts/manual_queries.json`. Seven regression checks cover split
grouping, tie handling, a training/checkpoint round trip and mode-specific feature
caching. The demo's mode selector, held-out protein examples and description
comparison also passed live checks.

## Data and splitting

`fetch.py` selects a reproducible shuffled subset of IDs from the existing
216,132-protein library and exports reviewed entries with both EC numbers and
function comments. Requests retrieve up to 100 accessions each; successful
batches are cached and resumable. Every response records the UniProt release,
URL, time and checksum. Mixed releases are rejected.

`prepare.py` verifies that the current Swiss-Prot sequence exactly matches the
original sequence used by Horizyn. It removes identical sequences, polyproteins,
uninformative names, sequences outside 50–1,500 residues, and descriptions
outside 15–250 words. Evidence and citation markers are removed from input text
but retained in the source records. Reviewed annotations include both
experimentally supported and inferred functions; the manifest records how many
have experimental function evidence.

All eligible sequences are clustered with MMseqs2 Linclust v1 (80 k-mers per
sequence) at 50% identity and 80%
coverage of both sequences, using connected components and explicit sequence
identity calculation and at most 128 rejected alignment candidates per sequence.
Components are further joined when their cleaned function
paragraphs are identical after punctuation/case normalization. Component IDs
deterministically assign splits. Sampling within each split then matches the
requested example counts, ordered across EC numbers. No protein, identical
sequence, exact function text, or detected cluster crosses splits. MMseqs2 is
heuristic: this is not a guarantee that every homolog has been separated.

Identical descriptions from different orthologs are retained within a split.
The number of unique descriptions is reported separately: matching the number
of training pairs does not match linguistic diversity or independent targets.

## Training and comparison

Two ridge heads (regularization 0.1 and 1.0) and a centered linear head are
compared using validation MRR. The linear head uses the reaction experiment's
cosine, contrastive and distillation weights (1, 0.5, 1), AdamW learning rate
0.002, batch size 256, and at most 64 epochs. Protein negatives and distillation
targets come from the training split. Identical function texts are treated as
multiple positives in the contrastive loss. The historic reaction checkpoint
used all reaction candidates as negatives, so this is not a completely matched
training-procedure control.

The winning head is selected without test results. Both heads are evaluated on
the same held-out Swiss-Prot descriptions, each with its own text instruction,
against the same frozen libraries. Reported measures include exact source
protein retrieval, proteins sharing a known reaction, and reaction retrieval.
Exact-protein retrieval is strict because a function paragraph often cannot
distinguish orthologs. Known reaction annotations are incomplete. Confidence
intervals resample whole split components; function-balanced recall gives equal
weight to each distinct function paragraph. Ties use average non-relevant rank.

Reverse transfer is measured on the original reaction test descriptions with
both heads. Neither pretrained Qwen nor Horizyn was retrained or made subject
to these new holdouts. Curated paragraphs are not a separate benchmark of
informal natural-language queries.

## Run

Run these from the repository root, with the existing teacher libraries and
reaction artifacts available. Install [MMseqs2](https://github.com/soedinglab/MMseqs2)
or pass `--mmseqs /path/to/mmseqs`; this run uses the downloaded binary in
`tools/mmseqs/bin/mmseqs` and records its version.

```bash
uv run python -m experiments.swissprot_poc.fetch --candidates 120000
uv run python -m experiments.swissprot_poc.prepare \
  --reference-fasta /path/to/original/horizyn/prots.fasta
uv run --extra modal modal run -m experiments.swissprot_poc.modal_run
uv run --extra modal modal run -m experiments.swissprot_poc.modal_run --task train
uv run python -m experiments.swissprot_poc.learning evaluate
uv run python -m unittest experiments.swissprot_poc.test_poc -v
uv run python -m web_demo.server
```

The Modal wrapper rents a single L4 with a one-hour timeout for frozen text
embedding and a 30-minute timeout for adapter fitting, with no automatic retries.
Evaluation runs locally. Local fitting is also supported with
`uv run python -m experiments.swissprot_poc.learning train --device mps`.
Outputs
are kept in the ignored `artifacts/` directory; completed datasets, feature
caches, training runs and evaluation reports refuse accidental replacement.
Downloading additional candidates reuses earlier batches.

The demo automatically discovers `artifacts/selected/adapter.pt` and enables
**Protein descriptions · Swiss-Prot**. Both heads share one local Qwen instance;
query caches are keyed by instruction and text. The existing reaction adapter
remains available. Protein results appear first in Swiss-Prot mode. Diagnostic
examples use the matching model's reserved pool.

## Sources

Annotations: [UniProtKB/Swiss-Prot](https://www.uniprot.org/help/uniprotkb_sections),
[batch downloads](https://www.uniprot.org/help/api_queries),
[CC BY 4.0 terms](https://www.uniprot.org/help/license).
Source evidence, accessions, exact sequence checksums and release information
are preserved under `artifacts/`.
