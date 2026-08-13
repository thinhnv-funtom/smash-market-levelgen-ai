"""Phase 1 (not yet implemented): level JSON <-> discrete relational token sequence.

Design (see smash-market-2/Doc/AILevelGenerator.md once it exists, or the planning
conversation this repo came from):

- Conditioning prefix: <BOS> <DIFF> <OBJCOUNT_BUCKET> <MOVECOUNT>
- Tables emitted first, assigned sequential ids — objects can only reference an
  id already emitted, so `tableId` validity is guaranteed by the grammar.
- Each object: <TYPE> <SIZE bucket, conditioned on TYPE via catalog.json>
  <RESTS_ON: table-surface | object-ref> (Y is derived at decode time from the
  anchor's top + this object's half-height — never a free token)
  <PLACE:grid + grid-cell coords | PLACE:freeform + continuous X/Z + yaw bucket>
- Canonical bottom-to-top emission order per table (raw JSON order is not signal).
- <EOS>

Needs data/catalog.json (Unity export) and a corpus (export_corpus.py) to build the
vocabulary against before this can be written for real — placeholder until then.
"""

raise NotImplementedError("Phase 1 — see module docstring and the plan.")
