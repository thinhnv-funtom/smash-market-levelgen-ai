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
  manifest.jsonl          # one line per corpus level: {path, source, checkpoint, approved_by,
                           # approved_at, excluded_reason}
  snapshots/              # frozen dataset_v{N}.jsonl compiled from the manifest at train time
  scratch/                # generate.py's default --output-dir when called from the Unity tool
src/levelgenai/
  export_corpus.py        # copies prod-13 JSON into data/corpus/prod13/, round-trip-excludes mismatches;
                           # --revalidate re-checks the existing manifest against the current vocab/quantizers
  manifest.py             # manifest read/write + snapshot compilation (AI-sourced fraction cap)
  promote.py              # add an approved generated level into data/corpus/ai_accepted/ + manifest
  stats.py                # corpus statistics (object/table/blocker counts, difficulty/moveCount dists)
  catalog.py               # loads catalog.json: per-type size variants + mass/friction/bounciness buckets
  geometry.py              # box/vector math (RESTS_ON inference, rotated extents)
  tokenizer.py             # level JSON <-> RelationalLevel (Phase 1 — table-first, RESTS_ON, type-conditioned size)
  roundtrip.py             # Phase 1 round-trip check, used by export_corpus.py's exclusion logic
  quantize.py              # uniform quantizers for the flat vocabulary's continuous fields
  vocab.py                 # the flat, finite token vocabulary (built from catalog.json)
  flatten.py               # RelationalLevel <-> flat integer token ids (Phase 2)
  flatten_check.py         # Phase 2 round-trip check, also wired into export_corpus.py
  model.py                 # nanoGPT-style decoder-only transformer
  train.py                 # training loop — tokenize once, stratified train/val split, AdamW, checkpoint
  generate.py              # samples from a checkpoint, runs validators.py, used by the Unity Editor tool
  validators.py            # structural / catalog / overlap / support / statistical-plausibility gates (Phase 3)
checkpoints/               # trained model weights (plain git while small; switch to git-lfs if the
                            # model-size sweep lands on something large — see below)
tests/                      # runnable directly, no pytest needed: PYTHONPATH=src python tests/test_*.py
```

## Design decisions (why things look like this)

See `smash-market-2`'s `Doc/AILevelGenerator.md` (the Unity Editor tool this repo's
`generate.py` is called from) and the planning conversation it came from for full context.
Summary:

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

## Training

Runs on a separate (e.g. GPU) machine — that's the whole point of this repo being
independent of the Unity checkout. Nothing here needs Unity or UnityEditor at all, only
`data/catalog.json` and `data/corpus/` (copy these two over from wherever you exported them,
or re-export directly if that machine also has network access to pull from the Unity repo).

### 1. Set up the environment

```
python -m venv .venv
.venv/bin/activate            # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` currently just pins `torch>=2.2` — install the CUDA build that matches
the machine's driver if you want GPU training (see pytorch.org's install matrix); `train.py`
auto-detects `cuda` vs `cpu` (`torch.cuda.is_available()`), no flag needed.

### 2. Make sure the data is there

```
ls data/catalog.json data/corpus/prod13 data/manifest.jsonl
```

If any are missing, either copy them from a machine that has the Unity checkout, or (if
this machine *does* have the Unity checkout too):

```
python -m levelgenai.export_corpus --unity-root /path/to/smash-market-2
```

### 3. Compile a training snapshot

```
python -c "
from pathlib import Path
from levelgenai.manifest import compile_snapshot
print(compile_snapshot(Path('.'), Path('data/manifest.jsonl'), Path('data/snapshots')))
"
```

Writes `data/snapshots/dataset_v1.jsonl` (or `_v2`, `_v3`, ... if one already exists —
snapshots are never overwritten, so every training run is tied to a specific, reproducible
one). Re-run this whenever `data/manifest.jsonl` changes (new hand-authored levels, newly
promoted generations) to pick up the additions.

### 4. Smoke-test before a real run

Confirm shapes/dtypes/loss all behave on a tiny model and a handful of steps — seconds, not
minutes, and catches most integration mistakes before they cost GPU time:

```
python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \
    --n-layer 2 --n-head 2 --d-model 32 --batch-size 4 --max-steps 20 --eval-interval 10
```

Expect `train_loss`/`val_loss` printed every `--eval-interval` steps and both trending down.
If it crashes, fix that before scaling up — a bug is far cheaper to find here than 15,000
steps into a real run.

### 5. Train for real

Model size is deliberately not defaulted for you (see `model.py`'s docstring: compute isn't
the bottleneck here, the ~791-level dataset is) — sweep a couple of sizes and compare
`val_loss`, don't just pick one and trust it:

```
python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \
    --n-layer 6 --n-head 6 --d-model 192 --batch-size 16 --max-steps 20000 --eval-interval 500
```

- Prints `train_loss`/`val_loss` every `--eval-interval` steps, and saves **two** checkpoints
  at each of those: `checkpoints/last.pt` (always — this is what `--resume` continues from)
  and `checkpoints/best.pt` (only on a new best `val_loss` — this is what `generate.py` should
  point at).
- `--lr` (default `3e-4`), `--val-fraction` (default `0.1`, stratified by difficulty — see
  `stratified_split`), `--seed` are also CLI flags if the defaults need adjusting.
- Watch for `val_loss` climbing back up while `train_loss` keeps falling — classic
  overfitting on a small dataset. If that happens fast, prefer a *smaller* model or more
  dropout (`--dropout`, default `0.1`) over a bigger one; see `model.py`'s docstring on why
  more parameters isn't automatically better here.
- There's no early stopping — `--max-steps` runs to completion regardless. Watch the log and
  Ctrl-C once `val_loss` has clearly stopped improving; `checkpoints/best.pt` already has the
  best checkpoint saved, so stopping early loses nothing.
- **Interrupted?** (crash, Ctrl-C, a Colab disconnect — see below) Resume from `last.pt`
  rather than restarting: `--resume checkpoints/last.pt` (the saved architecture wins, so
  `--n-layer`/`--n-head`/`--d-model`/`--dropout` are ignored when resuming — only `--max-steps`
  and similar training-schedule flags still apply).

### 6. Try generating from the checkpoint

Before wiring it into the Unity tool, confirm it produces *something* parseable:

```
python -m levelgenai.generate --checkpoint checkpoints/best.pt \
    --snapshot data/snapshots/dataset_v1.jsonl \
    --difficulty 0 --object-count-bucket 2 --move-count 24 --num-samples 8 \
    --output-dir data/scratch/manual_test
cat data/scratch/manual_test/summary.json
```

Early in training, expect a low accept rate (or 0) — the model hasn't learned the grammar
yet. A rising accept rate over successive checkpoints is a much more meaningful progress
signal than loss alone, since loss doesn't distinguish "nearly right" from "structurally
broken." Once this looks reasonable, point `Tools > Smash Market > AI Level Generator` at
the same `--checkpoint` path from Unity — see `Doc/AILevelGenerator.md` in the main repo.

## Training on Google Colab

Same steps as above, just run as notebook cells instead of a local shell — plus two
Colab-specific problems neither of which show up on a persistent machine: **the VM's local
disk is wiped every time the runtime disconnects**, and **free-tier sessions get disconnected**
(idle timeout around 90 minutes with no interaction; a hard cap around 12 hours regardless).
Plan around both rather than being surprised by them mid-run.

**1. Runtime → Change runtime type → GPU** (a T4 is plenty for the model sizes in step 5
above), then clone the repo — `data/catalog.json`, `data/corpus/`, and
`data/snapshots/dataset_v1.jsonl` are already committed, so nothing needs copying in:

```python
!git clone https://github.com/thinhnv-funtom/smash-market-levelgen-ai.git
%cd smash-market-levelgen-ai
!ls data/catalog.json data/corpus/prod13 data/snapshots/dataset_v1.jsonl
```

**2. Don't blindly `pip install -r requirements.txt`.** Colab ships a `torch` build already
matched to its CUDA driver; reinstalling a different one is how you end up with a `torch`
that can't see the GPU. Check first, and only install if this doesn't already look right:

```python
import torch
print(torch.__version__, "cuda:", torch.cuda.is_available())
```

**3. Mount Drive and checkpoint there, not to the VM's local disk** — `checkpoints/` inside
the cloned repo lives on the VM and is gone the moment the runtime resets, taking every
checkpoint with it:

```python
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/LevelGenAI/checkpoints'
```

**4. Smoke test** (same rationale as step 4 above — seconds, not minutes, before spending
real GPU time):

```python
!PYTHONPATH=src python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \
    --checkpoint-dir /tmp/smoke_test \
    --n-layer 2 --n-head 2 --d-model 32 --batch-size 4 --max-steps 20 --eval-interval 10
```

**5. Train, checkpointing to Drive:**

```python
!PYTHONPATH=src python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \
    --checkpoint-dir {CHECKPOINT_DIR} \
    --n-layer 6 --n-head 6 --d-model 192 --batch-size 16 --max-steps 20000 --eval-interval 500
```

**6. If the runtime disconnects mid-run**: reconnect, re-run cells 1–3 (clone + mount Drive —
`last.pt` is still on Drive even though the VM itself is fresh), then resume instead of
restarting from step 0:

```python
!PYTHONPATH=src python -m levelgenai.train --snapshot data/snapshots/dataset_v1.jsonl \
    --checkpoint-dir {CHECKPOINT_DIR} --resume {CHECKPOINT_DIR}/last.pt --max-steps 20000
```

**7. Sanity-check generation** (same as step 6 above):

```python
!PYTHONPATH=src python -m levelgenai.generate --checkpoint {CHECKPOINT_DIR}/best.pt \
    --snapshot data/snapshots/dataset_v1.jsonl \
    --difficulty 0 --object-count-bucket 2 --move-count 24 --num-samples 8 \
    --output-dir /content/scratch_test
!cat /content/scratch_test/summary.json
```

**7b. Generate a real batch here instead of locally.** `generate.py` has no KV-cache (see
`model.py`'s docstring) — every sampled token re-runs the forward pass over the whole context
so far, so a dense level (a big object-count bucket) is genuinely slow on a local CPU-only
`torch`. Colab's GPU makes this much cheaper, so do the actual candidate generation here, not
on whatever machine runs the Unity Editor:

```python
import itertools, json, zipfile
from pathlib import Path

OUT_DIR = "/content/batch_out"
combos = itertools.product([0, 1, 2], [1, 3, 5])  # (difficulty, object_count_bucket) — adjust freely

for difficulty, bucket in combos:
    out = f"{OUT_DIR}/d{difficulty}_b{bucket}"
    !PYTHONPATH=src python -m levelgenai.generate --checkpoint {CHECKPOINT_DIR}/best.pt \
        --snapshot data/snapshots/dataset_v1.jsonl \
        --difficulty {difficulty} --object-count-bucket {bucket} --move-count 24 \
        --num-samples 16 --output-dir {out}
    summary = json.load(open(f"{out}/summary.json"))
    print(f"d{difficulty} b{bucket}: {summary['accepted']}/{summary['requested']} accepted")

with zipfile.ZipFile("/content/batch_out.zip", "w") as zf:
    for path in Path(OUT_DIR).rglob("*.json"):
        zf.write(path, path.relative_to(OUT_DIR))
```

Download it (`from google.colab import files; files.download("/content/batch_out.zip")`), or
save straight to Drive instead if you'd rather not deal with a browser download. Either way,
what you get out is the same plain level JSON `generate.py` always produces — no Colab-specific
format. To get an accepted one into Unity, **you don't need the AI Level Generator window's
subprocess round-trip at all**: just point the existing, generic
`Tools > Smash Market > Level SO Generator` at a folder containing the `sample_*.json` files
whose `summary.json` entry says `"accepted": true` (check `rejections` for the rest — same
`support`/`overlap`/`catalog`/`plausibility`/`structural` reasons documented in the main repo's
`Doc/AILevelGenerator.md`). That window's import path is the exact same
`LevelSOGeneratorWindow.GenerateFromPaths` the Editor tool itself calls — it doesn't care
whether the JSON came from Unity's own subprocess call or a zip you downloaded from Colab.

**8. Get `best.pt` to wherever the Unity tool runs.** It's already durable on Drive — either
sync/download `best.pt` from Drive into the local checkout's `Tools/LevelGenAI/checkpoints/`,
or commit it to the repo from Colab so `git pull` picks it up elsewhere:

```python
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"
!cp {CHECKPOINT_DIR}/best.pt checkpoints/best.pt
!git add checkpoints/best.pt
!git commit -m "Add trained checkpoint from Colab"
```

Pushing needs a token — Colab has no stored GitHub credentials. Don't paste one in plaintext
into a cell (notebooks get shared/committed); read it interactively instead:

```python
from getpass import getpass
token = getpass("GitHub token: ")
!git push https://{token}@github.com/thinhnv-funtom/smash-market-levelgen-ai.git master
```

- `{CHECKPOINT_DIR}`/`{token}` interpolate the Python variable into the `!` shell command —
  that's Colab/IPython syntax, not a typo.
- A Personal Access Token (fine-grained, `contents: write` on just this repo) beats a
  password — GitHub no longer accepts password auth for git operations anyway.
- Checkpoints are small enough (the model sizes discussed above) to commit as plain git blobs
  — no LFS needed unless a later size sweep lands on something much bigger (see the Design
  decisions section above).

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
    the manifest rather than chased further. 791/1000 levels (79.1%) reached the clean
    training snapshot **before the floating-object fix below**; see that entry for the
    current number.
  - **Floating-object bug, found after the first real generation run** (not by testing
    against the corpus — the corpus round-tripped fine; this only showed up on model
    *output*): an object's X/Z were encoded as absolute `<COORD_*>` tokens completely
    independent of its `<ANCHOR_BACK_k>` choice. RESTS_ON guaranteed no *vertical* gap from
    whichever anchor was picked, but nothing tied X/Z to that anchor's footprint — a model
    could (and did) emit a valid anchor and a valid-but-unrelated position, producing an
    object whose Y sat on its anchor's top while its footprint floated somewhere else, or
    landed on a different object than the one Y was computed from ("touches only
    horizontally/diagonally, wrong logic" as reported). Root-cause fix: when the anchor is
    another object, X/Z are now encoded as an `<OFFSET_*>` **relative to that anchor's own
    x/z** (measured from the real corpus: 69.5% exactly `(0,0)`, p99 distance 2.0, max 3.2 —
    see `quantize.py`'s `OFFSET`), so "near its anchor" is a fact the model's own two choices
    produce together, not two independent fields that can silently disagree. Anchored-to-table
    objects are unaffected (still absolute `<COORD_*>`, unchanged).
    - This reintroduced the exact class of bug Phase 2 had already hit once for rotation:
      `OFFSET` is added onto an *already-decoded* (already-quantized) anchor position, so
      `Quantizer.decode()`'s normal bin-center convention has a constant +step/2 bias that
      **compounds additively down a RESTS_ON chain** (measured depth up to 11 in the real
      corpus) — an early version of this fix broke 99.7% of the corpus's round-trip before
      this was caught by testing against real data. Fixed with a dedicated
      `encode_grid`/`decode_grid` pair on `Quantizer` (nearest-grid-point instead of
      bin-center/floor) so an exact-multiple-of-`step` offset — the vast majority, since
      the real corpus already grid-snaps — round-trips with **zero** error instead of a
      biased one, and non-grid offsets get an unbiased ±step/2 instead of a one-directional
      bias. Used only for `OFFSET`; every other quantizer (`COORD`, `DIM`, `QUAT`, ...) is
      unchanged.
    - Net cost: this is a real, measured increase in excluded training levels, not a free
      fix — going from 791/1000 (79.1%) to **752/1000 (75.2%)** clean levels, all of the new
      exclusions being deep chains of genuinely non-grid-aligned (freeform/tilted) positions,
      the same class of thing Phase 1/2's existing exclusions already track. Re-run
      `python -m levelgenai.export_corpus --revalidate` (no `--unity-root` needed) after any
      future vocab/tokenizer/quantizer change — `export`'s `append_entry` is intentionally
      idempotent-by-path and will never re-check an already-recorded level on its own.
    - **Vocab size changed (950 -> 1014 tokens) — any existing checkpoint is now
      incompatible and must be retrained from scratch.** Token ids shifted, so loading an old
      checkpoint against the new vocab would silently mismatch embeddings, not just fail to
      load.
- **Phase 2 (model half)** — `model.py` (nanoGPT-style decoder-only transformer),
  `train.py` (loads a snapshot, tokenizes once, stratified train/val split by difficulty,
  dynamic per-batch padding, AdamW, checkpoints on best val loss) and `generate.py`
  (samples from a checkpoint, decodes back to level JSON, reports — doesn't crash on —
  any sample a still-learning model emits ungrammatically) are written but **NOT executed
  locally**: this machine's torch install is broken (missing `torchgen`), and per-turn
  decision, not worth fixing here since training happens on a separate GPU machine anyway.
  Smoke-test before a real run — see train.py's module docstring for the exact command
  (a few steps on a tiny model).
- **Phase 3 (validators)** — done: `validators.py`'s five gates (structural, catalog,
  overlap, support, statistical plausibility) run on generate.py's output before anything
  reaches Unity's import path. Verified against the real corpus (`tests/test_validators.py`,
  fully runnable now — no torch needed): structural/catalog/plausibility never reject real
  data (as expected, it's already valid); overlap+support together accept **665/752 (88.4%)**
  of the current clean set, consistent with the known, tolerated rotated-object corner
  clipping (see `Doc/LevelCoverage.md`'s warning-not-error philosophy) plus a bit of
  quantization noise on top.
  - **`support_check`** is the defense-in-depth backstop for the floating-object bug above:
    even though `OFFSET` makes "footprint overlaps the anchor" *likely*, nothing in the
    grammar makes it *certain* — a still-learning model can emit a valid anchor and a
    valid-but-large offset that place an object beside its anchor instead of on it. This
    check rejects exactly that: for every object anchored to another object, its XZ footprint
    must share AABB area with the anchor's — **the same AABB-overlap notion
    `geometry.py`'s `infer_anchors` uses to define RESTS_ON in the first place**
    (`xz_overlap_area`), not `overlap_check`'s stricter oriented-rectangle SAT test. That
    distinction mattered: an early version reused the stricter SAT test and flagged **425
    rejection instances across 46 real, already-clean levels** as "unsupported" — not a real
    bug, just disagreeing with the looser AABB rule that had already decided those were
    RESTS_ON relationships when the corpus was tokenized. Sharing the same overlap
    definition dropped that to **5 instances across 2 levels**, the tiny floating-point/epsilon
    residual you'd expect from checking real data against the exact rule that produced it.
  - Two real bugs found by this same test-against-real-data discipline: the overlap SAT
    check had an inverted epsilon sign that flagged *every touching pair* as overlapping
    (measured: 788/791 levels — obviously wrong for shipped content); and once fixed,
    `OVERLAP_EPS` needed widening from 0.05 to 0.15 to absorb the position noise the COORD
    quantizer itself introduces (two independently-quantized adjacent objects can close a
    real gap by up to ~0.125 units), or quantization alone pushed the false-positive rate
    from 15% to 31%.
