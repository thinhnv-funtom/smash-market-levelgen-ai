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
edits `prod13`'s own files, only this repo's copy. If `data/catalog.json` is present, it
also round-trip-checks every level (`roundtrip.py`) and marks any that don't decode back
exactly with `excluded_reason` in the manifest, so `compile_snapshot` leaves them out of
training without touching the source file. Currently **209/1000 levels excluded this way**
— see Status below for why.

## Status

- **Phase 0** — done: repo skeleton, corpus export, stats, manifest/promote plumbing.
- **Phase 1** — tokenizer/catalog/geometry done, with a known, measured, accepted gap:
  - Relational encode/decode (`tokenizer.py`) round-trips **97.5% of objects exactly**
    (83.8% of levels with zero mismatches) using symmetric per-type boxes (`geometry.py`) —
    table-first grammar, type-conditioned size, RESTS_ON instead of a free Y.
  - The residual ~2.5%: an object resting on a rotated support's actual sloped top face,
    not its flat symmetric top. Three approaches were tried and measured against the real
    corpus: (1) symmetric boxes — 2.5% mismatch (kept); (2) real collider pivot/bounds via a
    Unity export — regressed to 13.1% (the level-authoring tool assumes symmetric boxes, so
    real sub-cm collider noise fights that convention rather than fixing anything); (3) an
    unclipped tilted-plane contact-height formula — regressed further to 5.9% + dependency
    cycles (matches objects nowhere near the support's real footprint; needs the query point
    clipped to the support's actual rotated face rectangle to work, which is unimplemented).
  - Decision: keep (1), and **exclude the 162 affected levels from training** via the
    manifest mechanism above rather than keep chasing the geometry — revisit only if v1
    training results actually suffer from the smaller corpus.
- **Phase 2 (tokenization half)** — `quantize.py` / `vocab.py` / `flatten.py` done: the
  structural representation above flattens into a finite integer token sequence (`<BOS> <DIFF>
  <OBJCOUNT_BUCKET> <MOVECOUNT>`, table blocks, object blocks with a bounded `<ANCHOR_BACK_k>`
  back-reference instead of an absolute pointer) and back again, verified against the clean
  corpus (`flatten_check.py`, wired into `export_corpus.py`'s exclusion check same as Phase 1's).
  - Vocab size **950** tokens; max real sequence length **3611** (levels up to 408 objects).
  - Two real bugs found and fixed by testing against real data, not just written and trusted:
    independently-quantizing a rotation's 4 quaternion components doesn't reproduce a *unit*
    quaternion, and that small error **compounds down a RESTS_ON chain** (deep stacks drifted
    up to ~0.25 units) — fixed by giving the common **pure-yaw** case (~10% of objects) its own
    single-angle token instead of 4 independent components, which can't help but stay unit-norm.
    The angle formula itself then had a sign-canonicalization bug (`q` and `-q` are the same
    rotation; the naive half-angle formula could double past the token range and silently clamp
    to the wrong angle) — fixed by canonicalizing `w >= 0` before halving.
  - Residual gap after both fixes: **47 more levels** (on top of Phase 1's 162) fail only at
    the flatten layer — genuine multi-axis tilts (not pure yaw) still use the 4-component
    quaternion and can still compound over a deep chain. Same call as Phase 1: excluded via
    the manifest rather than chased further. **791/1000 levels (79.1%) reach the final clean
    training snapshot.**
- **Phase 2 (model half)** — `model.py` / `train.py` / `generate.py` / `validators.py` are
  still stubs; this is the next work.
