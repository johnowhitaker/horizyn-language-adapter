"""Group equivalent reactions before making deterministic train/validation/test splits."""

import argparse, csv, hashlib, random
from pathlib import Path
from data.io import read_jsonl, write_jsonl
from settings import RAW, PROCESSED
import json


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


def canonical_reaction(smiles):
    from rdkit import Chem

    sides = []
    for side in smiles.split(">>"):
        molecules = []
        for component in side.split("."):
            mol = Chem.MolFromSmiles(component)
            if mol is None:
                raise ValueError("Unparseable reaction")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            molecules.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        sides.append(".".join(sorted(molecules)))
    if len(sides) != 2:
        raise ValueError("Expected two sides")
    return ">>".join(sorted(sides))


def prepare(args):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    source = args.source
    with (source / "rhea_metadata.tsv").open() as f:
        metadata = {
            r["Reaction identifier"].split(":")[-1]: r
            for r in csv.DictReader(f, delimiter="\t")
        }
    with (source / "rhea_directions.tsv").open() as f:
        masters = {
            v: r["RHEA_ID_MASTER"]
            for r in csv.DictReader(f, delimiter="\t")
            for v in r.values()
        }
    rows = []
    rejected = []
    for split in ["train", "test"]:
        with (source / f"{split}_rxns.csv").open() as f:
            for row in csv.DictReader(f):
                rid = row["reaction_id"]
                master = masters.get(rid.split("_")[-1], rid.split("_")[-1])
                meta = metadata.get(master)
                if not meta:
                    rejected.append(
                        {"reaction_id": rid, "reason": "missing named equation"}
                    )
                    continue
                try:
                    canonical = canonical_reaction(row["reaction_smiles"])
                except ValueError:
                    rejected.append({"reaction_id": rid, "reason": "invalid smiles"})
                    continue
                rows.append(
                    {
                        "reaction_id": rid,
                        "rhea_master": master,
                        "reaction_smiles": row["reaction_smiles"],
                        "equation": meta["Equation"],
                        "ec": meta["EC number"],
                        "original_split": split,
                        "canonical": canonical,
                    }
                )
    # Connected components over structural equality, equation equality, and Rhea family.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        equation = "=".join(sorted(s.strip() for s in row["equation"].split("=")))
        for key in [
            ("structure", row["canonical"]),
            ("equation", equation),
            ("master", row["rhea_master"]),
        ]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["reaction_id"])
    for i, row in enumerate(rows):
        group = digest("|".join(sorted(groups[find(i)])))
        fraction = int(digest("horizyn-language-adapter:" + group)[:8], 16) / 2**32
        row["group_id"] = group
        row["split"] = (
            "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"
        )
        row["ec_class"] = (
            row["ec"].removeprefix("EC:").split(".")[0] if row["ec"] else "unknown"
        )
        del row["canonical"]
    # Round-robin EC classes, randomized within classes: broad coverage in early samples.
    buckets = {}
    for row in sorted(rows, key=lambda r: r["reaction_id"]):
        buckets.setdefault(row["ec_class"], []).append(row)
    rng = random.Random(20260905)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop())
    args.out.mkdir(parents=True, exist_ok=True)
    existing_path = args.out / "reactions.jsonl"
    if (args.out / "descriptions.jsonl").exists() and read_jsonl(
        existing_path
    ) != ordered:
        raise SystemExit(
            "Source records changed after generation; use a new output directory"
        )
    write_jsonl(existing_path, ordered)
    write_jsonl(args.out / "rejected.jsonl", rejected)
    manifest = {
        "version": 1,
        "rows": len(rows),
        "groups": len(groups),
        "rejected": len(rejected),
        "splits": {
            s: sum(r["split"] == s for r in rows)
            for s in ["train", "validation", "test"]
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.glob("*")
            if p.suffix in [".csv", ".tsv"]
        },
        "limitations": [
            "Exact structural/equation groups only; not a chemical-family holdout.",
            "Inference checkpoint saw released data. Test is held out from adapter only.",
            "Named-equation direction is used for prose; current reaction cache averages both directions.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=PROCESSED)
    prepare(p.parse_args())
