"""Training loop over a data/snapshots/dataset_v{N}.jsonl snapshot (see
manifest.compile_snapshot): tokenize every level once, split stratified by
difficulty, train a GPT (model.py) with next-token cross-entropy / teacher
forcing, log val loss, checkpoint the best.

NOT executed locally (see model.py's docstring — this machine's torch install
is broken). Before a real run on the training machine, smoke-test with:

    python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \\
        --max-steps 20 --eval-interval 10 --n-layer 2 --n-head 2 --d-model 32

...and confirm loss decreases and no shapes/dtypes error, before scaling up.
Model size is a CLI sweep, not a fixed default — see model.py's docstring on
why (compute isn't the bottleneck here, dataset size is).

Usage:
    python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \\
        --n-layer 6 --n-head 6 --d-model 192 --max-steps 20000

Saves two checkpoints every --eval-interval steps: last.pt (always, so a run
interrupted mid-training — e.g. a Colab disconnect, see README's Colab
section — never loses more than one eval interval) and best.pt (only on a
new best val_loss, so it's never worse than an earlier point in the same
run). Resume an interrupted run with --resume:

    python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \\
        --resume checkpoints/last.pt --max-steps 20000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from levelgenai.catalog import load_catalog
from levelgenai.flatten import to_tokens
from levelgenai.model import GPT, GPTConfig
from levelgenai.tokenizer import encode_level
from levelgenai.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sequences(snapshot_path: Path, catalog, vocab: Vocab) -> list[tuple[list[int], int]]:
    """Returns (token_ids, difficulty) per level — difficulty drives the
    stratified split below, since the corpus is 70/20/10 Easy/Medium/Hard
    (see the plan) and a random split could easily starve Hard in val.
    """
    sequences = []
    with snapshot_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            level = record["level"]
            ids = to_tokens(encode_level(level, catalog), vocab)
            sequences.append((ids, level["difficulty"]))
    return sequences


def stratified_split(sequences: list[tuple[list[int], int]], val_fraction: float, seed: int) \
        -> tuple[list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    by_difficulty: dict[int, list[list[int]]] = {}
    for ids, difficulty in sequences:
        by_difficulty.setdefault(difficulty, []).append(ids)

    train, val = [], []
    for group in by_difficulty.values():
        rng.shuffle(group)
        split_at = max(1, int(len(group) * val_fraction))
        val += group[:split_at]
        train += group[split_at:]
    return train, val


def make_batch(sequences: list[list[int]], batch_size: int, pad_id: int, device: str) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-batch padding (not a fixed block_size) — sequence length
    varies hugely here (16 to 3600+ tokens), so padding every batch to the
    global max would waste most of the compute on levels an order of
    magnitude shorter than the longest one.
    """
    batch = random.sample(sequences, min(batch_size, len(sequences)))
    max_len = max(len(s) for s in batch)

    x = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), max_len - 1), -1, dtype=torch.long)  # -1 = ignored by cross_entropy
    for i, seq in enumerate(batch):
        x[i, :len(seq) - 1] = torch.tensor(seq[:-1])
        y[i, :len(seq) - 1] = torch.tensor(seq[1:])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model: GPT, sequences: list[list[int]], pad_id: int, device: str,
              batches: int = 10, batch_size: int = 16) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = make_batch(sequences, batch_size, pad_id, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "catalog.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None,
                         help="Path to a last.pt/best.pt to continue from (see README's Colab section).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = load_catalog(args.catalog)
    vocab = Vocab(catalog)
    pad_id = vocab.id("<PAD>")

    sequences = load_sequences(args.snapshot, catalog, vocab)
    train_seqs, val_seqs = stratified_split(sequences, args.val_fraction, args.seed)
    block_size = max(len(s) for s in (train_seqs + val_seqs)) - 1
    print(f"train: {len(train_seqs)}, val: {len(val_seqs)}, block_size: {block_size}, device: {device}")

    start_step = 0
    best_val = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]  # the saved architecture wins — --n-layer etc. can't change mid-run
        model = GPT(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_val = checkpoint.get("best_val", checkpoint.get("val_loss", float("inf")))
        print(f"resumed from {args.resume} at step {start_step} (best_val={best_val:.4f})")
    else:
        cfg = GPTConfig(vocab_size=len(vocab), block_size=block_size, n_layer=args.n_layer,
                         n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
        model = GPT(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"model params: {model.num_params():,}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, args.max_steps + 1):
        x, y = make_batch(train_seqs, args.batch_size, pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = eval_loss(model, val_seqs, pad_id, device)
            print(f"step {step}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            best_val = min(best_val, val_loss)

            checkpoint = {"cfg": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "step": step, "val_loss": val_loss, "best_val": best_val}
            torch.save(checkpoint, args.checkpoint_dir / "last.pt")  # always — see --resume
            if val_loss == best_val:
                torch.save(checkpoint, args.checkpoint_dir / "best.pt")  # only on improvement — see generate.py

    print(f"done. best val_loss={best_val:.4f}, checkpoint at {args.checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
