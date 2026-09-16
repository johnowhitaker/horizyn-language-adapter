This repository implements the language-adapter experiment, not the Horizyn model.
Horizyn source and pretrained assets are fetched separately; retain and follow
Dayhoff Labs' PolyForm Noncommercial 1.0.0 license and the data/model terms:
https://github.com/dayhofflabs/horizyn

Qwen3-Embedding model cards identify their Apache-2.0 license. The optional
3Dmol.js viewer is loaded from a pinned release and is BSD-3-Clause licensed.
Protein metadata and structures come from UniProt and AlphaFold DB.
The Swiss-Prot experiment uses existing UniProtKB/Swiss-Prot annotations
(UniProt release recorded in the dataset manifest). UniProt database text is
provided under CC BY 4.0: https://www.uniprot.org/help/license . Function
paragraphs are cleaned of evidence tags and citation markers for model inputs;
the downloaded originals and accessions are retained in the experiment artifacts.

A license for this repository's own adapter code has not yet been selected.
