# 0.8B hybrid quantization quality pilot — 2026-09-30

**Update 2026-10-01:** a frozen data-aware amendment is reported in [GPTQ_PILOT.md](GPTQ_PILOT.md).

**Decision:** the tested weight-only PQ2_0 rules remain far below the FP source on this small natural-text screen. Do not claim functional quality replication or infer Prism's training recipe from this result. Continue with a separately frozen 1.7B reference comparison and data-aware reconstruction only if its resource and quality target justify it.

This pilot uses the pinned public `Qwen/Qwen3.5-0.8B` checkpoint as one unchanged FP ancestor. The [protocol](results/quality_pilot/protocol.json) fixes four public-domain books, two slices per book, the tokenizer, 128-token sequences, 64 target tokens, CPU runtime, and a 220,000,000-byte packed-file ceiling. *Dracula* and *Pride and Prejudice* were validation books; *Frankenstein* and *Sherlock Holmes* remained held out until [selection](results/quality_pilot/selection.json). The first [protocol version](results/quality_pilot/protocol_v1.json) covered absmax only; fixed-86 and least-squares formulas were amended before scoring those arms. The selection file was written before opening held-out scores.

The [native scorer](score_quality_pilot.cpp) uses the same model tokenizer for every arm, scores exact teacher-forced next-token probabilities at positions 64–127, and clears attention and recurrent memory between slices. The [analysis](analyze_quality_pilot.py) checks identical 128-token IDs for every paired slice, source/text hashes, script and protocol hashes, and the preselected arm. Full scores and model SHA-256 hashes are in [results.json](results/quality_pilot/results.json). This is a **descriptive two-book screen**, not a non-inferiority test with meaningful uncertainty coverage.

## Arms and results

Every packed arm has **151 PQ2_0, 36 BF16 recurrent-gate, and 133 F32 tensors**. The same source, roles, code layout, group axis, and precision exceptions are used across assignments. Base is unrotated; folded applies the independently checked signed H512 transform to FP weights *before* quantization. Absmax is the pinned native quantizer. [The assignment script](make_ls_pilot.py) creates the other two PQ2_0 payloads: fixed 86 largest magnitudes per 128-group with least-squares scale, and independent groupwise least-squares choice of support size and scale. Both use stable lower-index tie breaking and FP16 scale rounding. [Export validation](results/quality_pilot_validation.json) checks all 320 tensor types and dimensions for four new arms, exact source-to-packed bytes on 36 sampled groups, and byte identity of all 169 non-PQ2 payloads within each basis. The sampled groups decode identically in the pinned C and Python codecs; all generated GGUFs load and score in the native runtime.

| Arm | Validation NLL, nat/token | Packed file bytes |
|---|---:|---:|
| FP base | 3.0414 | 3,020,533,152 |
| FP H512 fold | 3.0414 | 3,020,564,704 |
| Base PQ2 absmax | 16.2932 | 213,700,000 |
| H512 PQ2 absmax | 16.0260 | 213,731,552 |
| Base PQ2 fixed-86 | 12.3406 | 213,700,000 |
| H512 PQ2 fixed-86 | 11.3003 | 213,731,552 |
| **Base PQ2 least squares** | **10.6614** | **213,700,000** |
| H512 PQ2 least squares | 12.3210 | 213,731,552 |

The frozen rule selected **base PQ2 least squares** by lowest mean validation NLL among packed arms. The held-out scores were FP **3.3839** and selected packed **10.2819** nats/token, a paired **+6.8981** nat/token loss. The two book-level differences were +6.9636 (*Frankenstein*) and +6.8326 (*Sherlock Holmes*). The file passed the 220 MB ceiling and missed the pre-held-out descriptive +0.05 nat/token tolerance by a wide margin. No alternative packed arm was scored on the held-out books.

## Interpretation and next gate

The H512 fold is function-preserving at FP precision on this pilot, but its benefit depends on the ternary assignment rule. Fixed-86 improved validation NLL relative to absmax in both bases; the lowest validation loss came from **unrotated** adaptive least squares. This does not establish a population ordering with two validation books. The large FP-to-packed gap is consistent with the earlier one-prompt logit smoke test and shows that merely exporting the correct runtime format does not recover source-model quality.

Before more expensive training, compare against the public Bonsai-1.7B reference under a frozen, matched evaluation. Expand independent books, add retrieval and feasible task metrics, and predeclare paired tolerances and byte budgets for that target. If a cheap method still fails, profile selective embedding/output or block reconstruction and then a bounded training run. This 0.8B model is a hybrid runtime bridge; it is not the Bonsai 2 27B ancestor or a quality proxy for its training result. The exact 27B FP-function gate remains open.
