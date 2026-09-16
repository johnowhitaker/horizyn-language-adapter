"""Fit frozen text features to frozen Horizyn vectors. Select on validation only."""

import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from settings import CACHE, RUNS
from training.model import Adapter
from training.common import load_data, ranks, metrics, device


DEFAULT = dict(
    kind="linear",
    hidden_dim=1024,
    center=True,
    cosine=1.0,
    contrast=0.5,
    distill=1.0,
    temperature=0.07,
    teacher_temperature=0.05,
    negatives="all",
    lr=0.002,
    weight_decay=0.01,
    batch_size=256,
    epochs=64,
    patience=6,
    seed=42,
)


def train(features, reactions, output, config, requested_device="auto"):
    if output.exists():
        raise FileExistsError(
            f"Run already exists: {output}. Choose a new output directory."
        )
    unknown = set(config) - set(DEFAULT)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    cfg = {**DEFAULT, **config}
    assert cfg["kind"] in ["linear", "mlp"] and cfg["negatives"] in ["all", "train"]
    assert cfg["epochs"] > 0 and cfg["batch_size"] > 0
    assert cfg["temperature"] > 0 and cfg["teacher_temperature"] > 0
    assert cfg["patience"] > 0
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    data = load_data(features, reactions)
    dev = device(requested_device)
    x = data["x"].to(dev)
    library = data["library"].to(dev)
    gold = data["gold"].to(dev)
    groups = data["groups"].to(dev)
    tr = torch.tensor(
        [i for i, r in enumerate(data["records"]) if r["split"] == "train"], device=dev
    )
    va = torch.tensor(
        [i for i, r in enumerate(data["records"]) if r["split"] == "validation"],
        device=dev,
    )
    assert len(tr) and len(va), "Both training and validation rows are required"
    output.mkdir(parents=True)
    candidates = (
        torch.unique(gold[tr])
        if cfg["negatives"] == "train"
        else torch.arange(len(library), device=dev)
    )
    model = Adapter(x.shape[1], library.shape[1], cfg["kind"], cfg["hidden_dim"]).to(
        dev
    )
    if cfg["center"]:
        model.center.copy_(x[tr].mean(0))
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    best = -1
    stale = 0
    history = []
    start = time.monotonic()
    for epoch in range(cfg["epochs"]):
        model.train()
        for ids in tr[torch.randperm(len(tr), device=dev)].split(cfg["batch_size"]):
            z = model(x[ids])
            target = library[gold[ids]]
            loss = cfg["cosine"] * (1 - (z * target).sum(-1)).mean()
            if cfg["contrast"]:
                logits = (z @ library[candidates].T) / cfg["temperature"]
                positive = groups[gold[ids], None] == groups[candidates][None, :]
                loss = (
                    loss
                    + cfg["contrast"]
                    * (
                        torch.logsumexp(logits, 1)
                        - torch.logsumexp(
                            logits.masked_fill(~positive, -float("inf")), 1
                        )
                    ).mean()
                )
            if cfg["distill"]:
                t = cfg["teacher_temperature"]
                teacher = (target @ library.T / t).softmax(1)
                loss = loss + cfg["distill"] * F.kl_div(
                    (z @ library.T / t).log_softmax(1), teacher, reduction="batchmean"
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schedule.step()
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == cfg["epochs"]:
            model.eval()
            with torch.inference_mode():
                result = metrics(ranks(model(x[va]), library, gold[va], groups))
            history.append(dict(epoch=epoch + 1, **result))
            print(json.dumps(history[-1]), flush=True)
            if result["mrr"] > best:
                best = result["mrr"]
                stale = 0
                checkpoint = dict(
                    state_dict={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    model_config=model.config,
                    encoder=data["encoder"],
                    teacher=data["teacher"],
                    config=cfg,
                    validation=result,
                    best_epoch=epoch + 1,
                    provenance=data["provenance"],
                )
                torch.save(checkpoint, output / "adapter.pt")
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
    report = dict(
        config=cfg,
        history=history,
        best_epoch=checkpoint["best_epoch"],
        validation=checkpoint["validation"],
        training_rows=len(tr),
        training_reactions=len(torch.unique(gold[tr])),
        elapsed_seconds=time.monotonic() - start,
        encoder=data["encoder"],
        teacher=data["teacher"],
        provenance=data["provenance"],
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=CACHE / "text.npz")
    p.add_argument("--reactions", type=Path, default=CACHE / "reactions.pt")
    p.add_argument("--out", type=Path, default=RUNS / "baseline")
    p.add_argument("--config", type=Path)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    train(
        a.features,
        a.reactions,
        a.out,
        json.loads(a.config.read_text()) if a.config else {},
        a.device,
    )
