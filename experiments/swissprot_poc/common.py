from pathlib import Path
import hashlib
import re
from settings import ENCODERS

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ENCODER = {
    **ENCODERS["qwen4b"],
    "prefix": "Instruct: Represent this protein function for retrieving matching proteins and biochemical reactions.\nQuery: ",
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def clean_function(text):
    text = re.sub(r"\{[^{}]*\}", "", text)
    text = re.sub(r"\((?:PubMed:[^)]*|By similarity)\)", "", text)
    text = re.sub(r"\bFUNCTION:\s*", "", text)
    text = re.sub(r"\s+([.,;])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    return re.sub(r"\s+", " ", text).strip(" ;.") + "."


def text_key(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def fasta_records(path):
    pid, pieces = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                if pid is not None:
                    yield pid, "".join(pieces)
                pid = line[1:].split()[0]
                if "|" in pid:
                    pid = pid.split("|")[1]
                pieces = []
            else:
                pieces.append(line.strip())
    if pid is not None:
        yield pid, "".join(pieces)
