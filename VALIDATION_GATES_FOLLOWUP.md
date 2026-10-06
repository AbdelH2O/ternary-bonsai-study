# Bonsai 2 validation gates: controlled follow-up decision

The planned fresh-document state assay has since completed. Its six-book result and updated decision are in the [fresh-state report](fresh_state/FRESH_STATE_REPORT.md); the next-work list below records the plan before that assay.

The initial [validation-gates report](VALIDATION_GATES_REPORT.md) established state save/restore feasibility and a six-prompt answer-margin assay. This follow-up executed the next small controls without changing the old frozen inputs or building a new perturbation matrix. The original natural-text held-out gate remains failed: alpha-minus-MLP long-history excess NLL was −0.000506 nat/token, its interval covered +0.005, and held-out short-history dose matching failed.

## Controlled state-import gate

The [state freeze](controlled_state/CONTROLLED_STATE_FREEZE.json) fixed the previously examined Pride and Prejudice span, 1K and 12K histories, original/alpha/MLP destinations, reciprocal recurrent-only and full-state imports, and an unrelated Dracula recurrent-state positive control. All direct scores matched the original sparse scorer exactly. All newly captured same-model full/recurrent roundtrips and all **18 A–B–A restores** had zero maximum per-token NLL difference. The unrelated-history control changed mean absolute continuation NLL by **0.881/0.884 nat/token** at 1K/12K, far above the frozen 0.01 floor. The [raw scores, hashes, and complete table](controlled_state/CONTROLLED_STATE_REPORT.md) document the method.

The pilot demonstrates a technically controlled, sensitive state-replacement assay. It does not identify alpha-gate damage: importing alpha recurrent state into original readout changed mean NLL by **−0.004403/−0.011588** nat/token (targets 1–95), while reciprocal, full-state, and MLP imports had different signs and magnitudes. The span and seed are not fresh, so these are descriptive method results. **Gate decision: go for a preregistered independent-span state assay; no mechanism claim from this pilot.**

## Expanded answer-margin gate

The [margin freeze](margin_extension/MARGIN_EXTENSION_FREEZE.json) fixed four new registry families (04–07), question 1, base/swap and three layouts: 24 correlated prompts, 96 original-model candidates with target-present/removal controls, then 48 present-target candidates per treatment arm. The exact GGUF chat prefix and nine target IDs per candidate are frozen. Original-model gold margins were positive on **24/24**, and removing the target association dropped the margin by at least 1 nat on **24/24**; mean margin moved from **+12.242** to **−12.343** nats. This passed the predeclared [baseline gate](margin_extension/baseline_gate.json).

| Checkpoint | Positive gold margins | Mean margin, nats | Matched far-minus-near margin, nats |
|---|---:|---:|---:|
| Original | 24/24 | +12.2419 | −0.8657 |
| Alpha seed 20260929 | 24/24 | +12.2105 | −0.8013 |
| MLP seed 20260929 | 24/24 | +12.2019 | −0.8175 |

The alpha-minus-MLP far/near change is **+0.0162 nats**, opposite the sign of extra alpha-specific far-distance margin damage. Across the four registry clusters its sign varies. These variants were not matched for retrieval-margin dose, and their prior held-out natural-text dose match failed. See the [full margin report](margin_extension/MARGIN_EXTENSION_REPORT.md) and [raw/paired analysis](margin_extension/variant_summary.json). **Gate decision: the assay is valid on these registries, but the one-seed treatment pilot gives no positive retrieval signal and does not warrant a larger retrieval perturbation matrix.**

## Next work

1. Freeze fresh independent text spans and a specific state-mediated contrast with a smallest meaningful effect, document-level uncertainty, and reciprocal/full/recurrent/A–B–A/positive controls. Compare models only after those identities pass on every new span. This is the highest-value next test of the recurrent-state mechanism.
2. If a retrieval claim remains important, extend the margin assay to fresh registries and **both** question targets, with base/swap, target-removal, and matched-length near/far pairs. Prespecify a registry-level threshold and sufficient independent families. Use accuracy as a companion result, since the earlier exact-answer pilot was at ceiling.
3. Revisit natural-text perturbation doses only after a broader, uncertainty-aware short-history equivalence calibration on new documents and a power calculation. Do not retune on the old four held-out books or infer an accumulated-state mechanism from permanent-weight loss alone.

The [state](controlled_state/ARTIFACTS.json) and [margin](margin_extension/ARTIFACTS.json) inventories hash the completed code, freezes, raw outputs, logs, and state files. CUDA inference used host-device execution because the default sandbox hides NVIDIA devices. The old frozen inputs and unrelated `.gitignore` and `uv.lock` changes were preserved.
