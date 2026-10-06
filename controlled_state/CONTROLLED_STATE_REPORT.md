# Controlled state-import method pilot

The [freeze](CONTROLLED_STATE_FREEZE.json) fixed one previously examined Pride and Prejudice span, 1,024/12,288-token histories, 96 identical targets, and original/alpha/MLP checkpoints (one perturbation seed each) before this run. The [runner](run.py) uses the pinned Prism CUDA build and the validated scorer's 512/512 prefill boundaries. The [raw comparison JSON](original-h1024-compare.json) and [summary](summary.json) contain per-token scores and source-state hashes for every arm and length.

**Technical controls passed.** All four newly captured MLP or unrelated-book source states had exact direct/full/recurrent roundtrips and exact agreement with prior scorer outputs. Each of the six destination-model/history direct scores matched the prior scorer exactly. All 18 A–B–A restores after imports returned exactly to the destination's direct per-token NLL. The imported state cannot affect target 0, which is scored from the destination history's final logit; target 0 was identical in every comparison. The unrelated Dracula recurrent-state positive control changed the original model's continuation by mean absolute **0.881** nat/token at 1K and **0.884** at 12K over targets 1–95, exceeding the frozen 0.01 sensitivity floor.

Mean imported-minus-direct NLL over the 95 state-affected targets (nat/token):

| Destination | Imported source/state | 1K | 12K |
|---|---|---:|---:|
| Original | Alpha recurrent only | −0.004403 | −0.011588 |
| Original | Alpha full | −0.005786 | −0.009193 |
| Alpha | Original recurrent only | +0.005037 | +0.007073 |
| Alpha | Original full | +0.004742 | +0.008190 |
| Original | MLP recurrent only | −0.000243 | −0.000275 |
| Original | MLP full | −0.000179 | +0.003179 |
| MLP | Original recurrent only | +0.004818 | −0.002351 |
| MLP | Original full | +0.000548 | −0.006136 |

These comparisons establish that state replacement is technically measurable and that readout/checkpoint direction and attention-KV inclusion matter. They do **not** establish an alpha-specific accumulated memory loss: the alpha-to-original import improves this one continuation, the reciprocal import changes sign, and MLP state imports also move token scores. A foreign checkpoint state can be off the receiving model's normal trajectory. The unrelated-history control demonstrates sensitivity, not a calibrated effect size for gate damage. The document, span, and seed were already used in earlier work, so this is a method pilot rather than an independent hypothesis test.

Next, freeze fresh independent spans and an explicit state-mediated estimand before treatment scoring. Include same-model roundtrips, A–B–A, unrelated-history positive controls, reciprocal and full/recurrent imports for each span. Use document-level uncertainty and treat the two directions as separate interventions; neither direction can be assumed to isolate a causal gate mechanism without additional compatibility checks. No broad perturbation matrix follows from this pilot alone.

Reproduce with `bash analysis/bonsai2/controlled_state/build.sh`, `python3 analysis/bonsai2/controlled_state/freeze.py`, then `python3 analysis/bonsai2/controlled_state/run.py` on host CUDA. The runner verifies source/binary, library, model, prior-freeze, token-ID, and state-file hashes before running. The existing frozen scorer and state-gate inputs were reused without modification.
