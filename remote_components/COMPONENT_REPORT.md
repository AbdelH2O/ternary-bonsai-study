# Remote-value state timing and R/S follow-up

This exploratory follow-up used the same eight registry pairs as the [completed remote-state gate](../remote_state_gate/REMOTE_STATE_REPORT.md). The earlier frozen +0.5-nat support gate failed; nothing here changes that decision. [Design](DESIGN.md), [freeze](COMPONENT_FREEZE.json), [runner](experiment.py), [raw comparisons](postquestion_comparisons/), [pre-question comparisons](prequestion_comparisons/), [summary](analysis_summary.json), and [artifact inventory](ARTIFACTS.json) retain the inputs and outputs. No weight variant was used.

**Technical gates passed.** Pinned Prism 842b188 writes 48 recurrent R tensor rows followed by 48 S rows for this model. Thirty-two R-only/S-only files were made by replacing payload bytes only; their headers and cell positions were unchanged. Own-state reconstruction and R+S counterpart reconstruction were byte exact. All 16 captures made immediately before `Question:` had exact full and recurrent-only roundtrip NLL vectors. In all 64 candidate comparisons, direct and own-state scores agreed, every A–B–A restoration agreed, and no position warning occurred. The post-question combined condition reproduced the earlier frozen raw comparison vectors exactly. The pre-question boundary is token aligned, 28–29 tokens before the answer boundary, and common within each base/swapped registry pair.

The estimand is the destination's own-state gold-minus-wrong complete-answer log-likelihood margin minus the margin after a counterfactual import. Positive means the destination's own state favors the correct association more. Base and swap destinations are averaged within registry; eight registries, not 16 destinations or hundreds of tokens, are the uncertainty units. Intervals are two-sided 95% t intervals with seven degrees of freedom.

| Import point and component changed | Mean effect, nat/answer | 95% t interval |
|---|---:|---:|
| After question: counterfactual R only | +0.035154 | [+0.012997, +0.057310] |
| After question: counterfactual S only | +0.227660 | [+0.057713, +0.397606] |
| After question: counterfactual R+S | +0.249476 | [+0.055474, +0.443478] |
| R/S interaction, combined minus component sum | −0.013337 | [−0.033427, +0.006753] |
| **Before question: counterfactual R+S** | **+0.006221** | **[−0.003585, +0.016027]** |

The post-question effect mostly follows **S**: the S-only mean is about 91% of the combined mean. R-only has a smaller positive effect. The components are not exactly additive, so the interaction is reported rather than discarded. The pre-question effect is about 2.5% of the post-question combined effect and is positive in six of eight registries, but its interval crosses zero. Its upper bound is below the frozen +0.1-nat threshold, yielding **exclusion of a +0.1-nat average pre-question state-margin effect in this fixed assay**. An independent calculation from the raw nine-token answer vectors reproduced every registry value in the [summary](analysis_summary.json).

| Registry | Post R only | Post S only | Post R+S | Pre-question R+S |
|---|---:|---:|---:|---:|
| 08 | +0.050025 | +0.117246 | +0.108780 | −0.004697 |
| 09 | +0.040776 | +0.243453 | +0.299903 | −0.011947 |
| 10 | +0.059174 | +0.177691 | +0.216186 | +0.001827 |
| 11 | +0.011831 | +0.104183 | +0.119139 | +0.015817 |
| 12 | +0.024075 | +0.084799 | +0.095315 | +0.008237 |
| 13 | −0.014135 | +0.029743 | −0.008040 | +0.005807 |
| 14 | +0.044171 | +0.447968 | +0.470604 | +0.026020 |
| 15 | +0.065311 | +0.616194 | +0.693921 | +0.008703 |

**Interpretation.** The earlier combined-state effect is a real effect of the imported post-question recurrent state in this assay, but this timing result does not support a meaningful effect of the state that existed before the question. The question can retrieve remote values from the destination's unchanged attention KV and write a value-dependent signature into later recurrent S. This is the leading explanation for why post-question S has a much larger effect. It is an inference, not a direct isolation of the attention pathway. The pre-question null could also reflect overwriting or masking when the question is processed with destination attention KV. Thus the result does not prove that DeltaNet lacks remote information, nor does it identify a quantization cause.

The earlier split changes CUDA prefill batch boundaries. Across the 32 candidates, the pre-question direct answer vectors differ from the old post-question direct vectors by up to 0.0571 nat on one token (mean of per-case maxima 0.0104). Each effect above compares imported and own states **within the same prefill profile**; the 2.5% cross-profile ratio is descriptive, not a tightly controlled timing effect.

**Go/no-go:** stop the planned alpha/MLP recurrent-state perturbation line at this gate. The existing registries have been used for design and mechanism, the post-question signal is small relative to the baseline answer margin, and the pre-question import excludes the chosen +0.1-nat effect in this fixed assay. A future test of genuine long-range recurrent retention would need fresh registries and a design that prevents attention from retrieving and rewriting the remote value during the question. A broad perturbation matrix is not warranted by these data.
