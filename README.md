# LevelGenAI

Training pipeline for a custom transformer that generates levels for **Smash Market**
(Unity project: `smash-market-2`), matching the style of its hand-authored level corpus.
Lives as its own repo so it can be checked out and trained **independently of the Unity
project** — on a separate (e.g. GPU) machine, without cloning the asset-heavy main repo.

Meant to be added to `smash-market-2` as a git submodule at `Tools/LevelGenAI`. The only
thing that ever crosses the boundary is plain JSON: this repo never touches `UnityEditor`
APIs, `.meta` files, or Unity binary assets.

## Layout

```
data/
  catalog.json            # exported from Unity (ObjectDatabaseSO) — see below. Not committed; regenerate.
  corpus/
    prod13/               # copy of Assets/_Use/Level/prod-13/*.json — frozen, never edit by hand
    manual/               # new hand-authored levels added later
    ai_accepted/          # generated levels that passed validation AND were played + approved
  manifest.jsonl          # one line per corpus level: {path, source, checkpoint, approved_by, approved_at}
  snapshots/              # frozen dataset_v{N}.jsonl compiled from the manifest at train time
src/levelgenai/
  export_corpus.py        # copies prod-13 JSON into data/corpus/prod13/ (one-time / on catalog change)
  manifest.py             # manifest read/write + snapshot compilation
  promote.py              # add an approved generated level into data/corpus/ai_accepted/ + manifest
  stats.py                # corpus statistics (object/table/blocker counts, difficulty/moveCount dists)
  tokenizer.py            # level <-> token sequence (TODO — Phase 1, see smash-market-2/Doc/AILevelGenerator.md)
  model.py                # transformer definition (TODO — Phase 2)
  train.py                # training loop (TODO — Phase 2)
  generate.py             # sampling / inference entry point used by the Unity Editor tool (TODO — Phase 2)
  validators.py           # structural / catalog / overlap / statistical-plausibility gates (TODO — Phase 3)
checkpoints/               # trained model weights (plain git while small; switch to git-lfs if the
                            # model-size sweep lands on something large — see the plan)
```

## Design decisions (why things look like this)

See `smash-market-2`'s `Doc/AILevelGenerator.md` (once Phase 4 lands) and the planning
conversation it came from for full context. Summary:

- **v1 scope**: objects + 1–5 tables, **no blockers** (rare in the corpus — 3.7% of levels —
  and only partially wired at runtime today).
- **Relational tokenization, not absolute coordinates**: tables are emitted first (so
  `tableId` references are valid by construction), and each object is placed via
  `RESTS_ON: table-surface | <other object>` rather than a free Y coordinate — this makes
  "no floating objects" true by construction instead of needing a physics check. Horizontal
  (X/Z) placement has two modes: `grid` (the common case, snapped positions) and `freeform`
  (a minority, correlates with non-identity rotation — confirmed against the real corpus).
- **Size is conditioned on type**: only look up the size buckets `catalog.json` actually
  lists for that `ObjectType` (mirrors `ObjectDefinitionSO.Variants`), never an
  independent/global size vocabulary.
- **Physics properties as conditioning context, not generation targets**: mass (per
  type+size, from each prefab's `Rigidbody`) and friction/bounciness (per material family,
  from the shared `PhysicsMaterial` assets) are exported into `catalog.json` and fed to the
  model as extra attributes on each object choice — they're fixed per type/size in the game,
  never authored per level instance, so the model only ever *reads* them.
- **Model size is not artificially capped**: training happens on a separate machine, so the
  real ceiling is dataset size (~1000 levels × ~4 after mirror augmentation), not compute.
  Sweep a few sizes and pick by validation loss, not a fixed parameter budget.
- **Data flywheel with a provenance guardrail**: `data/corpus/ai_accepted/` lets approved
  generations feed back into training (see `promote.py`), but the manifest always records
  provenance and `stats.py` should be re-run per snapshot so a drift away from the original
  human-authored distribution is visible before training on it, not after.

## Regenerating `catalog.json`

In the Unity project: `Tools > Smash Market > AI Level Generator > Export Catalog`, assign
the project's `ObjectDatabaseSO`, and export to `Tools/LevelGenAI/data/catalog.json` (the
default path already points here when this repo is checked out as the submodule).

## Regenerating the corpus

```
python -m levelgenai.export_corpus --unity-root /path/to/smash-market-2
```

Copies `Assets/_Use/Level/prod-13/*.json` into `data/corpus/prod13/` and seeds
`data/manifest.jsonl` with `source: human` entries for each. Safe to re-run — it never
edits `prod13`'s own files, only this repo's copy.

## Status

Phase 0 (this commit): repo skeleton, corpus export, stats, manifest/promote plumbing.
`tokenizer.py` / `model.py` / `train.py` / `generate.py` are stubs — Phase 1/2 next.
