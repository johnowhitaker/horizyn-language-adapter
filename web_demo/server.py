"""Serve the demo and local inference in one process. No paid API calls."""

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import requests
import torch
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from settings import ROOT, CACHE, PROCESSED, RUNS
from data.embed import TextEncoder
from data.io import read_jsonl
from training.model import load_adapter

lock = threading.RLock()
state = {"ready": False, "stage": "Loading local model"}
engine = None
checkpoint = Path(os.environ.get("HLA_CHECKPOINT", RUNS / "best/adapter.pt"))


class Engine:
    def __init__(self, path):
        torch.set_num_threads(4)
        self.adapter, self.checkpoint = load_adapter(path)
        rc = torch.load(CACHE / "reactions.pt", map_location="cpu", weights_only=True)
        assert rc["teacher"] == self.checkpoint["teacher"], (
            "Teacher libraries do not match adapter"
        )
        self.reactions = F.normalize(rc["embeddings"].float(), dim=-1)
        self.ids = rc["ids"]
        self.ri = {rid: i for i, rid in enumerate(self.ids)}
        self.metadata = rc["metadata"]
        self.groups = rc["groups"]
        self.proteins = None
        self.pids = []
        self.pi = {}
        if (CACHE / "proteins.pt").exists():
            pc = torch.load(
                CACHE / "proteins.pt", map_location="cpu", weights_only=True
            )
            assert pc["teacher"] == rc["teacher"]
            self.proteins = F.normalize(pc["embeddings"].float(), dim=-1)
            self.pids = pc["ids"]
            self.pi = {p: i for i, p in enumerate(self.pids)}
        self.protein_meta = (
            json.loads((CACHE / "protein_metadata.json").read_text())
            if (CACHE / "protein_metadata.json").exists()
            else {}
        )
        self.catalogue = None
        if (CACHE / "catalogue.npz").exists():
            with np.load(CACHE / "catalogue.npz") as f:
                m = json.loads(str(f["metadata"]))
                if m["encoder"] == self.checkpoint["encoder"]:
                    self.catalogue = F.normalize(
                        torch.tensor(f["embeddings"].astype("float32")), dim=-1
                    )
                    self.ci = torch.tensor(
                        [self.ri[r["reaction_id"]] for r in m["records"]]
                    )
        # Demo examples are explicit: serving the app never opens the test set automatically.
        self.examples = read_jsonl(PROCESSED / "demo_examples.jsonl")
        self.encoder = TextEncoder(self.checkpoint["encoder"], offline=True)
        self.query_cache = {}

    @torch.inference_mode()
    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.query_cache))
        for start in range(0, len(missing), 4):
            chunk = missing[start : start + 4]
            self.query_cache.update(zip(chunk, self.encoder(chunk)))
        return torch.stack([self.query_cache[t] for t in texts])

    def representation(self, texts, mode):
        x = self.encode(texts)
        if mode == "catalogue":
            if self.catalogue is None:
                raise HTTPException(
                    400,
                    "No catalogue for this encoder; embed the named equations first",
                )
            sim = x @ self.catalogue.T
            values, index = sim.topk(min(5, len(self.catalogue)))
            z = F.normalize(
                (
                    self.reactions[self.ci[index]]
                    * (values / 0.03).softmax(1)[:, :, None]
                ).sum(1),
                dim=-1,
            )
            scores = torch.full((len(texts), len(self.reactions)), -float("inf"))
            scores[:, self.ci] = sim
        else:
            z = self.adapter(x)
            scores = z @ self.reactions.T
        return z, scores

    def reaction_hit(self, i, score):
        r = self.metadata[i]
        rid = self.ids[i]
        return dict(
            id=rid,
            score=float(score),
            equation=r.get("equation", rid).replace(" = ", " → "),
            ec=r.get("ec", ""),
            url="https://www.rhea-db.org/rhea/"
            + str(r.get("rhea_master", rid.removeprefix("Rh_"))),
        )

    def enrich(self, pids):
        missing = [p for p in pids if p not in self.protein_meta]
        if not missing:
            return
        try:
            r = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params=dict(
                    query=" OR ".join("accession:" + p for p in missing),
                    format="tsv",
                    fields="accession,protein_name,organism_name,ec",
                    size=100,
                ),
                timeout=5,
            )
            r.raise_for_status()
            for row in csv.DictReader(r.text.splitlines(), delimiter="\t"):
                self.protein_meta[row["Entry"]] = dict(
                    name=row.get("Protein names", "").split(" (")[0],
                    organism=row.get("Organism", ""),
                    ec=row.get("EC number", ""),
                )
            (CACHE / "protein_metadata.json").write_text(json.dumps(self.protein_meta))
        except requests.RequestException:
            pass

    @torch.inference_mode()
    def search(self, text, mode="adapter", expected=None):
        started = time.monotonic()
        with lock:
            z, scores = self.representation([text], mode)
            scores = scores[0]
            seen = set()
            hits = []
            for i in scores.argsort(descending=True).tolist():
                if self.groups[i] in seen:
                    continue
                seen.add(self.groups[i])
                hits.append(self.reaction_hit(i, scores[i]))
                if len(hits) == 8:
                    break
            proteins = []
            if self.proteins is not None:
                values, indices = (self.proteins @ z[0]).topk(
                    min(8, len(self.proteins))
                )
                pids = [self.pids[i] for i in indices]
                self.enrich(pids)
                proteins = [
                    dict(
                        id=p,
                        score=float(s),
                        url="https://www.uniprot.org/uniprotkb/" + p,
                        **self.protein_meta.get(p, {}),
                    )
                    for p, s in zip(pids, values)
                ]
            result = dict(
                query=text,
                mode=mode,
                reactions=hits,
                proteins=proteins,
                seconds=time.monotonic() - started,
            )
            if expected:
                if expected not in self.ri:
                    raise HTTPException(400, "Target not in reaction library")
                i = self.ri[expected]
                group_scores = {}
                for g, s in zip(self.groups, scores.tolist()):
                    group_scores[g] = max(s, group_scores.get(g, -float("inf")))
                result["expected"] = dict(
                    target=self.reaction_hit(i, scores[i]),
                    rank=1
                    + sum(
                        s > group_scores[self.groups[i]] for s in group_scores.values()
                    ),
                    groups=len(group_scores),
                    exact_rank=1 + int((scores > scores[i]).sum()),
                )
        return result


def boot():
    global engine
    try:
        engine = Engine(checkpoint)
        state.update(ready=True, stage="Ready")
    except Exception as exc:
        state.update(stage="Could not load artifacts", error=str(exc))


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=boot, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


def ready():
    if engine is None or not state["ready"]:
        raise HTTPException(503, state)
    return engine


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    mode: str = Field(default="adapter", pattern="^(adapter|catalogue)$")
    expected: str | None = None


class Mode(BaseModel):
    mode: str = Field(default="adapter", pattern="^(adapter|catalogue)$")


class Comparison(Mode):
    protein_id: str
    descriptions: list[str] = Field(min_length=2, max_length=8)


@app.get("/api/health")
def health():
    return {
        **state,
        "reactions": len(engine.ids) if engine else 0,
        "proteins": len(engine.pids) if engine else 0,
        "catalogue": engine is not None and engine.catalogue is not None,
        "examples": bool(engine and engine.examples),
    }


@app.post("/api/search")
def search(body: Query):
    if not body.text.strip():
        raise HTTPException(422, "Enter a reaction description")
    return ready().search(body.text.strip(), body.mode, body.expected)


@app.post("/api/example")
def example(body: Mode):
    e = ready()
    if not e.examples:
        raise HTTPException(
            404, "No demo_examples.jsonl; choose diagnostic examples explicitly"
        )
    row = random.choice(e.examples)
    return e.search(row["text"], body.mode, row["reaction_id"])


@app.post("/api/compare")
@torch.inference_mode()
def compare(body: Comparison):
    e = ready()
    texts = [t.strip() for t in body.descriptions]
    if any(not t or len(t) > 3000 for t in texts):
        raise HTTPException(
            422, "Use 2–8 nonempty descriptions, each at most 3,000 characters"
        )
    if body.protein_id not in e.pi:
        raise HTTPException(404, "Protein not in library")
    with lock:
        z, _ = e.representation(texts, body.mode)
        scores = (z @ e.proteins[e.pi[body.protein_id]]).tolist()
    return sorted(
        [dict(text=t, score=s) for t, s in zip(texts, scores)],
        key=lambda r: -r["score"],
    )


@app.get("/api/structure/{pid}")
def structure(pid: str):
    if not re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", pid):
        raise HTTPException(400, "Invalid accession")
    path = CACHE / "structures" / (pid + ".pdb")
    if path.exists():
        return Response(path.read_text(), media_type="chemical/x-pdb")
    try:
        r = requests.get("https://alphafold.ebi.ac.uk/api/prediction/" + pid, timeout=8)
        r.raise_for_status()
        url = r.json()[0]["pdbUrl"]
        if urlparse(url).hostname != "alphafold.ebi.ac.uk":
            raise ValueError("Unexpected structure host")
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        if "ATOM " not in r.text:
            raise ValueError("No atoms")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return Response(r.text, media_type="chemical/x-pdb")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        raise HTTPException(404, "No structure available")


app.mount("/", StaticFiles(directory=ROOT / "web_demo/static", html=True), name="demo")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=checkpoint)
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    checkpoint = a.checkpoint
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=a.port)
