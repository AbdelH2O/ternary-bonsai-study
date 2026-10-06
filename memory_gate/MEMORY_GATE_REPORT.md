# Original-model recurrent-memory gate: result and decision

The [frozen design](DESIGN.md) compares the original checkpoint's own 1K and 12K recurrent R/S states while keeping its weights, 12K attention KV, 96 target IDs, and scoring profile fixed. It reuses twelve saved spans from six books. [Version 2 of the freeze](MEMORY_GATE_FREEZE_V2.json) retains the same numerical rule and records a [technical amendment](TECHNICAL_AMENDMENT.md) after version 1 stopped on its first case: pinned Prism checks and applies a recurrent slot separately, so the deliberately unrebased 1K state emits **two** identical position warnings, not one. The original stopped [freeze](MEMORY_GATE_FREEZE.json), program, partial JSON, and log remain intact; no first-case statistic was used to change the windows or thresholds.

**Every version-2 technical gate passed.** For all twelve comparisons, the direct score, own-12K-state import, and every A–B–A restoration matched the saved independent reference exactly (maximum per-token NLL difference **0**). Raw 1K and position-rebased 1K imports also matched exactly on all 96 target scores (maximum difference **0**). The rebased files differ from their source only in the four-byte recurrent-cell position field, verified byte-for-byte; each raw import produced the two expected warnings and each rebased import produced none. Target 0 was computed before the import and excluded from the effect. The [freeze](MEMORY_GATE_FREEZE_V2.json), per-case [results](results_v2/), source hashes, and [summary](analysis_summary.json) preserve the evidence. An independent recomputation from all twelve raw comparison JSONs reproduced the book means and intervals below.

Memory benefit is `NLL(1K R/S; 12K attention KV) − NLL(12K R/S; 12K attention KV)`. Positive values mean the longer-built recurrent state improves prediction under this fixed receiver context. The two spans are averaged within each book; the six books define uncertainty.

| Book | Targets 1–16, nat/token | Targets 1–95, nat/token |
|---|---:|---:|
| Emma | +0.088997 | +0.040120 |
| Little Women | +0.037476 | +0.086053 |
| Wuthering Heights | +0.102238 | +0.041357 |
| The War of the Worlds | +0.238480 | +0.084645 |
| The Count of Monte Cristo | +0.042026 | +0.050897 |
| Middlemarch | −0.025122 | +0.027031 |
| **Six-book mean** | **+0.080682** | **+0.055017** |
| **95% t interval, df 5** | **[−0.013169, +0.174534]** | **[+0.029099, +0.080935]** |

**Frozen decision: unresolved.** The primary 1–16 mean exceeds the declared +0.005 floor and five of six books are positive, but its lower 95% bound is below zero. Its upper bound is above +0.005, so the gate neither supports nor excludes that effect by its predeclared rule. The 1–95 secondary window is positive in all six books with a positive interval. That is useful evidence that longer-built R/S state changes natural-text prediction over the continuation, but the secondary result cannot override the primary decision.

The contrast isolates state construction history while holding the receiver checkpoint and attention KV fixed. It does **not** isolate DeltaNet S from convolution memory R, and importing a 1K state beside 12K attention KV creates a hybrid context the model did not encounter in ordinary prefill. The exact raw/rebased-score agreement addresses position metadata in this pinned path; it does not remove that hybrid-context limitation. The six books and reused positions also limit generalization. Runtime R/S storage remains F32, so this is a history-content intervention, not a runtime precision experiment.

The next informative work is a separately frozen sensitivity test on new independent documents or a task with an identical recent suffix and a query demonstrably dependent on remote information. If a primary gate then establishes recurrent benefit, split S-only and R-only with identity and restoration controls before comparing alpha/MLP perturbations. The current result does not authorize a broad perturbation matrix.

Reproduce the frozen result in order with `python3 analysis/bonsai2/memory_gate/gate_v2.py freeze` in a fresh copy without an existing freeze, then `python3 analysis/bonsai2/memory_gate/gate_v2.py run` on **host CUDA**, then `python3 analysis/bonsai2/memory_gate/gate_v2.py analyze`. The runner rejects changed frozen inputs and validates every comparison before analysis. The [artifact inventory](ARTIFACTS.json) records the completed files and hashes. The prior assay inputs, checkpoints, and unrelated `.gitignore`/`uv.lock` changes were not modified.
