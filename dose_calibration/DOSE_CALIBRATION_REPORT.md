# Natural-text dose calibration and corrected retrieval baseline

**Decision:** Freeze the alpha-only 20% scale perturbation and a 20%-coverage, 20%-scale MLP control across three sign seeds for a **small perturbation retrieval pilot**. Both produce modest, similar short-history damage on the calibration set. The beta-only intervention does not reproducibly damage that set. The corrected retrieval baseline passes answer-swap and length audits, but its 48/48 accuracy is at ceiling; perturbation sensitivity and the held-out long-context contrast remain untested. No held-out treatment scores or full perturbation matrix were run.

## Frozen inputs and measurement

The [calibration manifest](calibration_manifest.json) freezes four new Project Gutenberg books, distinct from the scorer-validation/held-out books, with source and full token-ID hashes. Two 96-ID spans per book begin at positions 30,000 and 50,000. The same scorer consumes `[p-1024:p]` and scores `[p:p+96]` with no BOS or chat template. Two baseline runs on these eight spans were bitwise identical (maximum per-token NLL difference 0). The original GGUF hash remained `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.

The [variant builder](build_variant.py) made Btrfs reflinks and left the source read-only. It changes `ssm_alpha.weight` or `ssm_beta.weight` separately in all 48 recurrent blocks, or the FP16 scale of evenly spaced `ffn_up.weight` PQ2_0 groups at a specified coverage. Each tensor has balanced positive/negative signs shuffled by seed. The builder verifies every realized rounded value and every byte outside intended patch ranges; its manifests record candidate SHA-256, selected groups, and changed values. The scorer used the same 16,384-context, one-sequence, f16-KV, flash-attention, 512/512 CUDA profile for every arm. The [per-condition JSONL](calibration_results.jsonl) keeps all gold-token NLLs, input/patch hashes, timing, settings, and errors; [all sweep means](sweep_summary.csv) are also saved.

Seed `20260929` established that alpha damage increased from +0.0013 at 5% to +0.0037 at 10% and +0.0069 at 20% mean excess NLL. Beta at 10% and 20% improved the mean for that seed. MLP strength and coverage were varied independently: 5% coverage at 20%, 30%, and 40% scale change yielded +0.0018, +0.0034, and +0.0050; 10% coverage at 20% yielded +0.0024; 20% coverage at 20% and 40% yielded +0.0057 and +0.0163. The 5%-coverage/40% MLP setting changed sign across seeds, so it was rejected in favor of the broader, lower-strength control.

| Seed | Alpha 20% | Beta 20% | MLP coverage 20%, scale 20% |
|---|---:|---:|---:|
| 20260929 | +0.006875 | −0.004629 | +0.005660 |
| 20260930 | +0.004670 | −0.000183 | +0.002522 |
| 20260931 | +0.003851 | +0.005164 | +0.007948 |
| Mean | **+0.005132** | +0.000117 | **+0.005377** |

Values are excess gold NLL in nats per target token versus the identical baseline target. All three alpha and selected MLP seed means are positive, between +0.002 and +0.02, and their mean doses differ by 0.000245 nat/token. A descriptive bootstrap resampling books and sign seeds gives 95% ranges of +0.00125 to +0.00829 for alpha, +0.00075 to +0.01091 for MLP, and −0.00549 to +0.00627 for MLP minus alpha. Dose selection used these same four books, so these intervals are not independent confirmation. The [dose freeze](DOSE_FREEZE.json) records selected model/patch hashes, seeds, the calibration analysis hash, the held-out scorer manifest hash, and the pilot IDs. Copies of the six selected patch manifests are kept in [selected_patch_manifests](selected_patch_manifests/). It does **not** freeze a held-out effect threshold or authorize a full matrix.

## Baseline retrieval gate

The frozen pilot used four registries from the corrected [retrieval-v2 prompts](../prompts/retrieval_v2.jsonl): two questions, base/value-swap counterparts, and short-near/long-near/long-far layouts for each, **48 requests** total. The original model returned the exact requested code in **48/48** requests. All **24 base/swap pairs** changed response to the correct swapped value. Long-near and long-far had exactly matched server token counts for each pair; the server's counts matched the frozen GGUF chat-template counts on every request. There were no request errors. Mean wall time was 1.21 seconds short-near, 9.86 seconds long-near, and 9.96 seconds long-far. See the [pilot summary](retrieval_pilot_summary.json) and [raw responses](retrieval_baseline_pilot.jsonl).

The baseline verifies answer tracking and avoids the old unique-prefix/name-format shortcuts, but 48/48 accuracy provides no measure of how much perturbation would be needed to cause a behavioral failure. A small, matched alpha/MLP perturbation retrieval pilot should test sensitivity before any full held-out matrix. If accuracy remains at ceiling, answer likelihood needs a validated way to score *both* correct and wrong responses, or the task needs harder interference while preserving exchangeability and the near/far length match. Natural-text calibration itself is not evidence of recurrent-state accumulation.

To reproduce the analysis from the saved raw runs, run `python3 analysis/bonsai2/dose_calibration/analyze.py` and `python3 analysis/bonsai2/dose_calibration/analyze_retrieval_pilot.py`. To reproduce inference, use host-device CUDA execution; the default sandbox hides `/dev/nvidia*`. The prior frozen primary, calibration-v2, and LongBench selection inputs and unrelated `uv.lock` modification were left untouched.
