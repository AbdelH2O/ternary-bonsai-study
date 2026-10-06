# Lesson 8 — Reproducibility and execution

> **In one sentence:** fix every input, program and decision rule *before* looking at treatment results, fingerprint everything, and measure how much the numbers move from execution details alone.

[← Lesson 7](07-state-import.md) · [Course home](README.md) · [Next: statistics →](09-statistics-and-claims.md)

---

## 8.1 Pinning and hashing

- **Pin:** use one specific Prism llama.cpp release (here `842b188`), not "whatever is installed".
- **SHA-256 hash:** a byte-level fingerprint. Change a single byte and the hash changes completely. The runs hash the model, tokenizer, scorer binary, linked CUDA libraries, input texts, token-ID files and patches. Matching hashes are evidence that nothing changed between the freeze and the run.
- **Artifact inventory:** a hash list of the final outputs.
- **JSONL:** one JSON result per line. Keeping **raw per-token vectors** lets anyone recompute the headline numbers independently. 🟢 Several reports did exactly this and reproduced their means and intervals.

⚠️ Hashes make an experiment **auditable**. They do not make its design **valid**.

## 8.2 Freeze (preregistration) and amendments

A **freeze** records inputs, estimands, thresholds and analysis code *before* treatment results are examined. It stops you from choosing, after the fact, the window or threshold that makes the result look best.

A **versioned technical amendment** repairs an implementation problem while keeping the failed version and explaining the change. 🟢 Example: the [memory gate](../memory_gate/MEMORY_GATE_REPORT.md) v1 stopped on its first case because the runtime emitted **two** position warnings instead of the expected one. v2 fixed the check, kept the same numerical rule, and left v1's files intact. No first-case statistic was used to change thresholds.

## 8.3 The execution profile, and why batch size changes scores

Common scoring profile: 16,384-token context, one sequence, CUDA offload, F16 attention KV, flash attention, logical/physical batch 512/512.

- **Logical batch:** the group of tokens submitted for evaluation.
- **Physical microbatch:** a smaller unit the backend may execute internally.

🟢 Changing the physical microbatch from 512 to 256 kept the token IDs fixed but **changed some NLLs**. GPU floating-point sums done in a different order round differently. Consequences:

- Comparisons need a **fixed execution profile** and a measured **numerical floor**.
- Even moving a capture boundary can change CUDA batching and therefore the direct scores. 🟢 The pre-question profile differed from the older post-question profile by up to **0.0571 nat** on one token. So import effects were compared **within** a profile, never across profiles.

The scorer also checked that state resets between documents, persists across evaluation batches, and reproduces exactly on rerun.

## 8.4 Host-device CUDA

On this workstation the default sandbox hides `/dev/nvidia*`. **Host-device CUDA execution** gives the pinned binary GPU access. It is an *execution requirement*, not a treatment condition. CPU-only work can still tokenize, build prompts, audit shortcuts, parse state files and hash artifacts.

Timings, VRAM use, load errors and position warnings are all kept, because a missing or failed case can bias a result if it is silently dropped.

---

## Check yourself

1. Two runs have identical SHA-256 hashes for every input. Does that make the experiment's conclusion correct?
2. Why can't you compare a direct score from the pre-question run with an import score from the post-question run?
3. After a freeze, you notice a bug that crashes the first case. What's the honest procedure?

<details><summary>Answers</summary>

1. No. It shows the inputs didn't change. The design can still be flawed; the first pilot's shortcut, for example, would hash perfectly.
2. Different CUDA batching produces per-token differences up to 0.057 nat, which could swamp or fake an import effect. Compare within one profile.
3. Write a versioned technical amendment that fixes the implementation, keep the failed version's files, keep the numerical rule unchanged, and don't use any outcome from the failed case to adjust choices.

</details>
