import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_records(rows):
    """Accept nested generated descriptions, flat text rows, or named equations."""
    result = []
    for row in rows:
        base = {k: row[k] for k in ["reaction_id", "group_id", "split"]}
        descriptions = row.get(
            "descriptions",
            [
                {
                    "text": row.get("text", row.get("equation")),
                    "style": row.get("style", "equation"),
                }
            ],
        )
        for d in descriptions:
            result.append({**base, **d})
    return result
