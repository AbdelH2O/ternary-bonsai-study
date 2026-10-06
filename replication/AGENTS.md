# Agent guide — Bonsai replication experiments

This directory holds the user's independent Bonsai replication research (see [REPLICATION_PLAN.md](REPLICATION_PLAN.md)). Follow the frozen-design discipline of the existing experiments: designs and protocols are frozen and approved before any GPU run, and every result report leads with its frozen decision.

## Recording every step

The user wants every step recorded so the work carries proof and a logical line of reasoning that can be shared or shown as a portfolio (requested 2026-10-06). [RESEARCH_LOG.md](RESEARCH_LOG.md) is the chronological record.

- **Add an entry for every step that produces evidence or a decision,** in the same session, before moving on. Steps include experiments, probes, forensics, design versions, approvals, incidents and abandoned attempts.
- **Entry shape:** date and title; the question (quote the user when the user asked it); what was done, with links to scripts; evidence, with links to reports and result files and SHA-256 prefixes for artifacts later claims depend on; outcome with evidence tags; the decision, who made it, and their words if the user made it.
- **Record failures and incidents as carefully as results:** wrong turns, misreadings corrected later, crashed runs, re-runs and their reasons. A reader should be able to see why each next step followed from the last.
- **Append only.** Don't rewrite past entries. A correction is a new entry that points back.
- **Keep detail in the reports.** The log links to them and keeps the chain of reasoning.
- **Version designs.** A design doc changed after review gets a version number and a revision-history row, with links to the evidence that motivated the change.
- **User approvals** are quoted verbatim, and bound to file hashes when a frozen design or protocol exists, as the existing `*_approval.json` records do.
- **Commit each step.** `analysis/bonsai2/` is its own git repository (separate from the Bonsai-demo fork). After writing a log entry, commit the entry together with the files it refers to, with a message naming the step. Tag frozen designs and protocols (for example `qat17b-design-v1`). Large artifacts stay ignored and are referenced by hash. The repository is public (<https://github.com/AbdelH2O/ternary-bonsai-study>), so push each step to `origin` after checking that the commit contains no credentials, no personal budget figures and no employer references; personal planning stays in gitignored files. Never force-push or rewrite published history.

## Renting GPU compute (guideline agreed with the user, 2026-10-06)

**Which provider for which job:**

1. **Modal first.** H100 SXM costs $3.95/h. Use it for learning-rate probes, dry runs and the main training run of each experiment, since it costs nothing out of pocket. Launcher: [modal_qat17b.py](modal_qat17b.py), volume `bonsai-qat17b`.
2. **Lium for work beyond what Modal covers.** Lium (lium.io) is a Bittensor Subnet 51 marketplace, with H100 SXM from about $1.30/GPU-h, billed per second. Use it for extension runs, extra runs, or teacher generation if the local GPU is too slow. RunPod or Vast.ai are fallbacks under the same rules.
3. **Local RTX 5070 (12 GB)** for data prep, teacher generation, export and evaluation. It's free. The bb sandbox hides the GPU, so GPU commands are run by the user or as unsandboxed systemd user units.

**Rules for any marketplace host (Lium, Vast, RunPod community):**

- **Treat each machine as disposable.** Hosts are independent operators and can disappear mid-run.
- **No secrets on the box.** Without confidential-computing (CVM) isolation, the provider may be able to access the pod. Never copy Modal, Hugging Face write, GitHub or other credentials there. Only public models, public data and public code go on it.
- **Keep resume states where the host can't take them.** A local volume dies with its machine. Use the provider's backup feature (test a restore first) or sync each resume state off the host. A 1.7B save is latents 5.6 GB plus optimizer state 5.6 GB; save every 20–40 minutes.
- **Check every new pod before real work.** Confirm `nvidia-smi` shows the GPU you paid for (for example H100 80GB HBM3, not PCIe). Then run the short profile (`qat_17b.py profile`, about 10 cents) and proceed only if step time is within about 10% of the reference: 2.63 s/step on Modal H100 SXM, micro-batch 8 × 4, no gradient checkpointing, cached ternary weights.
- **Check payment and refund terms** before topping up. How renters pay, and what is billed when a pod disappears, wasn't verified for Lium as of 2026-10-06.
- **Use a provider-neutral launcher.** The Modal launcher doesn't port. Use a container with the pinned stack (torch 2.14.0, transformers 5.5.4), input and checkpoint sync, and a resume-on-start entrypoint.

**Prices:**

- Price aggregators such as [computeprices.com](https://computeprices.com/gpus/h100) are fine for a shortlist, not as the source of record. They're scraped, carry referral links, and their coverage is incomplete.
- Confirm the price on the provider's own page on the day you rent.
- Budget by measured hours × confirmed price per finished run, not by price per hour. PCIe and power-limited cards are cheaper per hour but slower.
