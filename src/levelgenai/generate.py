"""Sampling entry point the Unity Editor tool shells out to (see the plan's
Phase 4 / Doc/AILevelGenerator.md). Given a requested difficulty/size/
moveCount profile, samples N candidate token sequences and runs each through
validators.py (Phase 3) — only ACCEPTED levels are written to --output-dir,
so Unity never has to re-implement or re-run any of those checks itself. A
summary.json alongside them (accepted/rejected counts + rejection reasons)
is what the Editor tool's preview list reads.

NOT executed locally (see model.py's docstring). Smoke-test on the training
machine against a checkpoint from a short train.py run before trusting output.

Usage:
    python -m levelgenai.generate --checkpoint checkpoints/best.pt \\
        --snapshot data/snapshots/dataset_v1.jsonl \\
        --difficulty 0 --object-count-bucket 2 --move-count 24 --num-samples 8 \\
        --output-dir /tmp/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from levelgenai.catalog import load_catalog
from levelgenai.flatten import build_prefix
from levelgenai.model import GPT
from levelgenai.validators import StatsProfile, validate
from levelgenai.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def sample_token_rows(model: GPT, vocab: Vocab, difficulty: int, object_count_bucket: int,
                       move_count: int, num_samples: int, temperature: float, top_k: int,
                       device: str) -> list[list[int]]:
    prefix = build_prefix(vocab, difficulty, object_count_bucket, move_count)
    idx = torch.tensor([prefix] * num_samples, dtype=torch.long, device=device)
    eos_id = vocab.id("<EOS>")
    out = model.generate(idx, max_new_tokens=model.cfg.block_size - len(prefix),
                          eos_id=eos_id, temperature=temperature, top_k=top_k)
    # Trim anything past <EOS> — validators.validate() rejects a row with none
    # at all (a truncated generation), so leaving the tail in changes nothing
    # for it, but a trimmed sequence is what flatten.from_tokens actually expects.
    return [row[:row.index(eos_id) + 1] if eos_id in row else row for row in out.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "catalog.json")
    parser.add_argument("--snapshot", type=Path, default=REPO_ROOT / "data" / "snapshots" / "dataset_v1.jsonl",
                         help="Used to fit validators.StatsProfile's plausibility bounds.")
    parser.add_argument("--difficulty", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--object-count-bucket", type=int, required=True, choices=range(8))
    parser.add_argument("--move-count", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = load_catalog(args.catalog)
    vocab = Vocab(catalog)
    stats = StatsProfile.from_snapshot(args.snapshot)
    model = load_model(args.checkpoint, device)

    rows = sample_token_rows(model, vocab, args.difficulty, args.object_count_bucket,
                              args.move_count, args.num_samples, args.temperature, args.top_k, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted = 0
    results = []
    for i, ids in enumerate(rows):
        result = validate(ids, catalog, vocab, stats)
        entry = {"sample": i, "accepted": result.accepted, "rejections": result.rejections,
                  "warnings": result.warnings}
        if result.accepted:
            file_name = f"generated_{i}.json"
            (args.output_dir / file_name).write_text(json.dumps(result.level), encoding="utf-8")
            entry["file"] = file_name
            accepted += 1
        results.append(entry)

    summary = {"requested": args.num_samples, "accepted": accepted,
               "rejected": args.num_samples - accepted, "samples": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"accepted {accepted}/{args.num_samples} — wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
