# Remote-value state sensitivity: result and decision

The [frozen design](DESIGN.md) used the untouched corrected retrieval registry families **08–15**. Earlier GPU pilots used 00–07; these eight families were fresh to inference. In each family, a base and value-swapped `long-far` prompt changed two remote code assignments while keeping the question and final **1,024 token IDs identical**. Both candidate answers were scored as the same nine-token complete-code-plus-end-of-turn continuation under the pinned GGUF chat template. The [freeze](REMOTE_STATE_FREEZE.json) fixed all prompt, token-ID, executable, library, and model hashes before inference.

The baseline sensitivity gate passed: the original model preferred the correct code in **16/16** base/swapped prompts; gold-minus-wrong complete-answer margins ranged from **+7.212 to +13.350 nats** (mean +11.092). The code identity reversed within each value-swap pair. Thus these prompts require the remote assignment rather than a changed recent suffix, although baseline accuracy alone is saturated.

All state-method gates passed. Sixteen same-model captures matched the independent split-prefill baseline score exactly, and both full-state and recurrent-only roundtrips had **zero** maximum per-token NLL difference. For 32 candidate comparisons, direct scores, own-state imports, and every A–B–A restoration likewise matched exactly. All source states had equal cell positions within their registry pair; no non-consecutive-position warning appeared. Target 0 was computed before import and is not an independent state-swap check. The [baseline scores](baseline.jsonl), [baseline gate](baseline_gate.json), [capture gate](capture_gate.json), per-case [captures](captures/) and [comparisons](comparisons/) retain raw nine-token vectors and hashes.

With the destination's original weights and attention KV held fixed, the primary effect is its own-state gold-minus-wrong margin minus the margin after importing the value-swapped counterpart's recurrent R/S state. Base and swap effects are averaged within each registry, making **eight registries** the uncertainty units.

| Registry | Own minus counterfactual margin, nat/answer |
|---|---:|
| 08 | +0.108780 |
| 09 | +0.299903 |
| 10 | +0.216186 |
| 11 | +0.119139 |
| 12 | +0.095315 |
| 13 | −0.008040 |
| 14 | +0.470604 |
| 15 | +0.693921 |
| **Eight-registry mean** | **+0.249476** |
| **95% t interval, df 7** | **[+0.055474, +0.443478]** |

The frozen rule required a mean of at least **+0.5 nat**, a positive lower interval bound, and six positive registries for support. Seven registries were positive, and the interval excludes zero, but its upper bound is **below +0.5**. The predeclared decision is therefore **exclusion of a +0.5-nat average state-margin effect in this fixed assay**, with evidence for a smaller positive effect. Independent recomputation from the raw comparison JSONs reproduced the mean, interval, and seven positive registries. All counterfactual-state margins remained positive; no answer preference flipped. The effect mostly changed the wrong-code score: its counterfactual-minus-own NLL averaged −0.24938 nat, while the gold-code change averaged +0.000095 nat.

This is evidence that the combined R/S state carries some remote-value influence even when the destination attention KV holds its own value, but the measured influence is small compared with the mean +11.092-nat baseline margin. The mixed state is off the model's normal joint attention/recurrent trajectory; the assay cannot assign a causal share to DeltaNet S versus convolution memory R or establish a general bound over prompts and checkpoints. Eight registry clusters and one original checkpoint limit generalization. A broad alpha/MLP perturbation matrix is not justified by this gate. If effects below +0.5 nat are scientifically worth pursuing, the next work is an independently frozen replication or a harder remote-information task with less saturated correct-code likelihood; a focused S-only/R-only split would then test mechanism before weight treatments.

Reproduce the phases with `python3 analysis/bonsai2/remote_state_gate/gate.py freeze` in a clean copy without a freeze, then `baseline`, `captures`, `comparisons`, and `analyze` in order. The split-prefill scorer is built from pinned Prism source by [build.sh](build.sh). GPU phases require host-device CUDA because the default sandbox hides `/dev/nvidia*`. The [artifact inventory](ARTIFACTS.json) hashes the completed files. Earlier frozen inputs, model weights, and unrelated `.gitignore`/`uv.lock` changes were left untouched.
