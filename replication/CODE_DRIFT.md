# Code drift: how far training moved the ternary codes — 2026-10-05

Descriptive, CPU-only, read-only over frozen checkpoints; not part of any sealed protocol. Script: [code_drift_probe.py](code_drift_probe.py); data: [results/code_drift/drift.json](results/code_drift/drift.json). Evidence tags: 🟢 measured; 🟡 inferred.

**Question.** An HF forensics thread reports that ~92% of Bonsai 2 27B codes equal rotate-then-absmean-RTN of the ancestor. If Prism's small models were equally "light-touch", a cheap recipe might exist. Do our students move as much as Prism's training did?

**Metrics** (parameter-weighted over all ternary projections; embedding excluded):
- *agree RTN*: codes equal to absmean round-to-nearest of the ancestor in the quantization basis (H512 folded for ours, unrotated for Prism gen-1 1.7B);
- *support changed*: zero/nonzero status differs from top86 of the ancestor;
- *sign flipped*: both nonzero, opposite sign;
- *nz sign agree*: nonzero codes whose sign matches the ancestor weight;
- *rel. weight change*: ‖dequant(codes) − W‖ / ‖W‖ (basis-invariant).

## 🟢 Results

| Model | Tokens | Agree RTN | Support changed | Sign flipped | Nz sign agree | Rel. weight change |
|---|---:|---:|---:|---:|---:|---:|
| Ours, top86 of ancestor (no training) | 0 | 97.0% | 0 | 0 | 100% | 0.455 |
| qat_42 (plain text), t8m / t16m / t33m | 8M / 16M / 33M | 92.8 / 89.8 / 87.1% | 6.7 / 9.8 / 12.6% | ≈0 | 99.96 / 99.8 / 99.5% | 0.46 / 0.47 / 0.48 |
| qat_42 (plain text), final | 65.5M | 86.3% | 13.5% | 0.03% | 99.3% | 0.487 |
| C qat_chat_42, final | 65.5M | 86.1% | 13.6% | 0.04% | 99.3% | 0.487 |
| M (MC data), final | 65.5M | 86.1% | 13.6% | 0.04% | 99.3% | 0.487 |
| U (trained FP tensors), t33m | 33M | 86.9% | — | — | 99.4% | — |
| **Prism Ternary-Bonsai-1.7B** (gen 1) | unknown | **60.9%** | **36.1%** | **2.9%** | **86.8%** | **1.00** |

Per kind, Prism moves q/k most (nz sign agree 0.84) and v/o least (0.88–0.91); ours moves ffn_down/ffn_up most (support changed 0.18/0.16) and attn_v least (0.09).

## 🟡 Reading

1. **At 1.7B Prism's training was not light-touch.** About 39% of codes differ from plain rounding, matching ParetoQ's "~40% of weights move below 2 bits" reconstruction regime. The 92% figure is for Bonsai 2 27B (a later generation, rotated, 16× larger) and does not transfer to small models.
2. **Our students move ~14% after 65.5M tokens, and the movement is decelerating** with the cosine schedule (6.7 → 9.8 → 12.6 → 13.5%). Nearly all of it is zero↔nonzero swaps of marginal weights; outright sign flips are 0.04% (Prism 2.9%). The latents themselves move little (‖ΔW_t‖/‖W‖ = 0.19).
3. **Data and trained FP tensors do not change which codes move**: plain text, chat, MC data and U are indistinguishable on every aggregate.
4. Movement is necessary, not sufficient, for quality, and the comparison crosses model size (0.8B vs 1.7B), generation and basis. Still, it supports the token-budget/step-size explanation over a missing cheap trick.

**Implication.** Either far more tokens or larger effective steps on the latents. The only latent-LR evidence is a 120-step probe (`work/qat08/probe`: monitor KL 1.26 at 2e-5, 1.13 at 1e-4, 2.80 at 5e-4), and short probes penalize high LRs. So a full-length higher-LR arm, judged on monitor KL *and* drift at equal tokens, is the cheapest untested lever.
