# ProtNLM2 protein-description experiment

Status: skipped after source reconnaissance on 2026-09-15. No training,
paraphrase generation, paid compute, or demo integration was started.

## Scope and stopping condition

The proposed experiment maps frozen Qwen3-Embedding-4B text features to frozen
Horizyn protein vectors, adding a protein-description adapter to the demo.
The user approved a separate experiment directory, approximately 10,000
proteins at most, preference for enzymes, and the previously proposed $25
experiment ceiling.

The user also required a bulk source of precomputed annotations and explicitly
ruled out collecting the dataset through thousands of per-accession annotation
requests. No usable public bulk source was found, so implementation stopped.
Do not resume per-accession collection under this approval.

## Download findings

- The official [ProtNLM2 FTP directory](https://ftp.ebi.ac.uk/pub/contrib/UniProt/ProtNLM2/)
  contains one accession-list TSV. It has 26,856 entries and seven columns:
  accession, entry name, organism ID/name, protein names, gene names, and length.
  It does not contain the predicted function descriptions.
- [UniProt's help page](https://www.uniprot.org/help/ProtNLM) explicitly says
  those protein names are existing UniProt names, not ProtNLM2 predictions, and
  that the new predictions are stored separately from ordinary UniProtKB records.
- The public [ProtNLM controller](https://github.com/ebi-uniprot/uniprot-rest-api/blob/main/uniprotkb-rest/src/main/java/org/uniprot/api/uniprotkb/controller/ProtNLMUniProtKBController.java)
  exposes JSON retrieval for one accession at a time. No bulk prediction route
  was found in that controller.
- A separate research group's [data history](https://github.com/ai4curation/ai-gene-review/blob/main/projects/PROTNLM_EVALUATION/data_history.md)
  describes an earlier 28,553-entry XML export, but supplies no public download
  for it. Their current retrieval procedure uses individual UniProt API calls.
  The derived TSV/JSONL datasets are excluded from their Git repository, their
  published releases have no attached assets, and the corresponding hosted
  prediction TSV/JSONL paths returned 404. A Hugging Face dataset search for
  `protnlm` returned no datasets.

These checks establish that a usable bulk source was not found; they do not
establish that no such export exists anywhere.

## Enzyme filtering and scale

A local scan of the official accession-list contents found **4,371 entries**
with an EC annotation matching `\(EC [1-7]\.` in the existing protein-name field.
This is a conservative candidate filter available without annotation calls.
It is not a complete enzyme census, nor a count of entries with predicted
functional prose. If a bulk export becomes available, intersect its functional
descriptions with this subset first; catalytic GO evidence could extend it.

Earlier reconnaissance, before the bulk-only condition, inspected two documented
examples and 64 randomly selected entries. Of the random sample, 22 contained
functional prose. This small sample should not be treated as a full coverage
count. No per-accession prediction requests were made after the bulk-only
condition was introduced.

## Existing assets and possible resumption

The existing reaction experiment has 58,710 text pairs, including 46,845 training
pairs. Its candidate libraries contain 11,797 reactions and 216,132 proteins.
The ProtNLM2 accession list had zero exact accession matches to that protein
library, so new sequence embeddings would be needed.

If a suitable bulk export is provided, resume within this directory, cap the
protein cohort at 10,000, and prefer the enzyme subset. Preserve the export's
version and evidence provenance. Use ProtT5 sequence features followed by the
frozen Horizyn protein encoder, checking compatibility against existing vectors
before scaling. The neighboring `horizyn-putida/demos/embed_custom_proteome.py`
contains a reusable starting point for sequence embedding.

Keep homologous sequences and duplicate text out of opposing train/evaluation
splits. Compare the new adapter against the existing reaction adapter on the
same candidate library, and distinguish exact-protein retrieval from functional
equivalence. Adding the demo selector remains pending a trained and evaluated
checkpoint.
