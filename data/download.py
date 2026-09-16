"""Fetch public Horizyn source/data and Rhea labels. No credentials required."""

import argparse
import gzip
import hashlib
import shutil
import subprocess
import urllib.parse
import urllib.request
from settings import RAW

UPSTREAM = "https://github.com/dayhofflabs/horizyn.git"
REVISION = "6944198303f2f0946d448a259ab788589cfd27b3"
FILES = {
    "train_pairs.csv": "d77c894783a2d3552b90b26eb253633b",
    "test_pairs.csv": "cdfab924e78d86b35adfcd7c01700974",
    "train_rxns.csv": "7b0335ac694e4afee87e7a0a970f56e4",
    "test_rxns.csv": "a45305ba22d4077d7a3f07d5f5d93ff5",
}


def download(url, path, md5=None, header=None):
    if path.exists():
        if md5:
            with path.open("rb") as f:
                actual = hashlib.file_digest(f, "md5").hexdigest()
            if actual != md5:
                raise RuntimeError(f"Checksum mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as r, temp.open("wb") as f:
        shutil.copyfileobj(r, f)
    if md5:
        with temp.open("rb") as f:
            actual = hashlib.file_digest(f, "md5").hexdigest()
        if actual != md5:
            raise RuntimeError(f"Checksum mismatch: {url}")
    if header:
        with temp.open("rb") as f:
            assert f.readline().startswith(header), "Unexpected table header"
    temp.replace(path)
    print("Downloaded", path, flush=True)


def main(proteins=False):
    for name, md5 in FILES.items():
        download(
            f"https://zenodo.org/api/records/17957034/files/{name}/content",
            RAW / name,
            md5,
        )
    download(
        "https://zenodo.org/api/records/20348783/files/horizyn_v1_0_inf.ckpt/content",
        RAW / "horizyn_v1_0_inf.ckpt",
        "cf6775b775287462099ae0681485a6bc",
    )
    params = urllib.parse.urlencode(
        dict(
            query="status:approved",
            columns="rhea-id,equation,ec,go,uniprot",
            format="tsv",
            limit="20000",
        )
    )
    download(
        "https://www.rhea-db.org/rhea/?" + params,
        RAW / "rhea_metadata.tsv",
        header=b"Reaction identifier\tEquation",
    )
    download(
        "https://ftp.expasy.org/databases/rhea/tsv/rhea-directions.tsv",
        RAW / "rhea_directions.tsv",
        header=b"RHEA_ID_MASTER\tRHEA_ID_LR",
    )
    upstream = RAW / "horizyn"
    if not upstream.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", UPSTREAM, str(upstream)], check=True
        )
        subprocess.run(["git", "-C", str(upstream), "checkout", REVISION], check=True)
    if (upstream / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        actual = (upstream / ".revision").read_text().strip()
    if actual != REVISION:
        raise RuntimeError("Upstream checkout revision changed")
    if proteins and not (RAW / "prots_t5.h5").exists():
        zipped = RAW / "prots_t5.h5.gz"
        download(
            "https://zenodo.org/api/records/17957034/files/prots_t5.h5.gz/content",
            zipped,
            "eaf845701188e52e50abab1a239c0d34",
        )
        tmp = RAW / "prots_t5.h5.tmp"
        with gzip.open(zipped, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        with tmp.open("rb") as f:
            assert (
                hashlib.file_digest(f, "md5").hexdigest()
                == "282cf3f6e7a502d98ece793d366e75e9"
            )
        tmp.replace(RAW / "prots_t5.h5")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proteins", action="store_true")
    main(p.parse_args().proteins)
