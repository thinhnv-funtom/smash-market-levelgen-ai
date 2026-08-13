"""Phase 2 (not yet implemented): decoder-only GPT-style transformer over the
tokenizer.py vocabulary.

Model size is deliberately not fixed here — training happens on a separate
machine, so the real ceiling is dataset size (~1000 levels x ~4 after mirror
augmentation), not compute. train.py should sweep a few sizes and pick by
validation loss, not a hardcoded parameter budget.
"""

raise NotImplementedError("Phase 2 — see module docstring and the plan.")
