"""Sampling entry point the Unity Editor tool shells out to (see the plan's
Phase 4). Given a requested difficulty/size/moveCount profile, samples N
candidate token sequences, decodes each back into level JSON, and writes
whatever parses — malformed sequences (a model still learning, or an unlucky
sample) are reported and skipped here, not treated as a crash. Real
acceptance (catalog/overlap/statistical-plausibility gates) is validators.py,
Phase 3 — this only guarantees the output is well-formed JSON, not good.

NOT executed locally (see model.py's docstring). Smoke-test on the training
machine against a checkpoint from a short train.py run before trusting output.

Usage:
    python -m levelgenai.generate --checkpoint checkpoints/best.pt \\
        --difficulty 0 --object-count-bucket 2 --move-count 24 --num-samples 8 \\
        --output-dir /tmp/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from levelgenai.catalog import load_catalog
from levelgenai.flatten import build_prefix, from_tokens
from levelgenai.model import GPT
from levelgenai.tokenizer import decode_level
from levelgenai.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_levels(model: GPT, vocab: Vocab, catalog, difficulty: int, object_count_bucket: int,
                     move_count: int, num_samples: int, temperature: float, top_k: int, device: str) \
        -> tuple[list[dict], list[str]]:
    prefix = build_prefix(vocab, difficulty, object_count_bucket, move_count)
    idx = torch.tensor([prefix] * num_samples, dtype=torch.long, device=device)

    eos_id = vocab.id("<EOS>")
    out = model.generate(idx, max_new_tokens=model.cfg.block_size - len(prefix),
                          eos_id=eos_id, temperature=temperature, top_k=top_k)

    levels, failures = [], []
    for i, row in enumerate(out.tolist()):
        try:
            ids = row[:row.index(eos_id) + 1] if eos_id in row else row  # trim anything past EOS
            rel = from_tokens(ids, vocab, catalog)
            levels.append(decode_level(rel, catalog, level_index=0))
        except Exception as e:  # a still-learning model can emit an ungrammatical sequence
            failures.append(f"sample {i}: {type(e).__name__}: {e}")
    return levels, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "catalog.json")
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
    model = load_model(args.checkpoint, device)

    levels, failures = generate_levels(
        model, vocab, catalog, args.difficulty, args.object_count_bucket, args.move_count,
        args.num_samples, args.temperature, args.top_k, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, level in enumerate(levels):
        (args.output_dir / f"generated_{i}.json").write_text(json.dumps(level), encoding="utf-8")

    print(f"wrote {len(levels)} levels to {args.output_dir}")
    if failures:
        print(f"{len(failures)} samples failed to parse:")
        for f in failures:
            print(f"  {f}")


if __name__ == "__main__":
    main()
