# Technical amendment after the first-case stop

The first frozen run stopped after one case because the runner expected one `non-consecutive token position` warning for a raw 1K state in a 12K continuation and Prism emitted two. No six-book statistic was computed. The original `DESIGN.md`, `gate.py`, `MEMORY_GATE_FREEZE.json`, and stopped first-case log/partial JSON remain intact.

The pinned source calls `llama_memory_recurrent::find_slot` twice: `prepare()` checks a candidate and restores the cell metadata, then `llama_memory_recurrent_context::apply()` applies it. Both calls log the same warning for the deliberately unrebased 1K state. A one-case diagnostic with only raw and rebased imports confirmed two raw warnings; the saved raw and position-rebased 96-token NLL vectors were exactly equal. The rebase changes only bytes 12–15 (one cell position, 1023 to 12287); the runner verifies every other byte is identical.

Version 2 changes **only** the expected warning count from one to two, and stores new comparisons under `results_v2/`. The books, spans, model, states, windows, thresholds, and other technical gates are unchanged. The two warnings must both identify the source position 1023 and destination continuation position 12382. Freeze version 2 before the full run. The first-case diagnostic is a method repair, not independent evidence about the six-book effect.
