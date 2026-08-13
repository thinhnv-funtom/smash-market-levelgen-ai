"""Phase 3 (not yet implemented): accept/reject gates for a generated level, run
before it's ever handed to Unity's LevelSOGeneratorWindow import path.

1. Structural — sequence decodes into well-formed table/object blocks (catches
   decode-level malformation; table-id validity and "resting" are already
   guaranteed by the tokenizer's grammar, not re-checked here).
2. Catalog validity — every (type, size) pair is in catalog.json (defensive
   re-check against catalog drift; the tokenizer already restricts to this set).
3. Geometric validity — port the SAT overlap test from
   Assets/@SmashMarket/Scripts/Level/Editor/LevelDataSOEditor.Overlap.cs.
4. Statistical plausibility — reject object-count/moveCount/difficulty combos far
   outside stats.py's observed joint distribution; flag (not reject) any table
   with 0 objects.
"""

raise NotImplementedError("Phase 3 — see module docstring and the plan.")
