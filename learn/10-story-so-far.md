# Lesson 10 — The story so far and what comes next

> **In one sentence:** each stage either fixed a flaw in the previous one or narrowed the question. The current evidence points away from "gate damage accumulates with distance" and toward "attention retrieves the fact at question time", and the next experiment is designed to tell those apart.

[← Lesson 9](09-statistics-and-claims.md) · [Course home](README.md)

---

![Timeline](img/timeline.svg)

## 10.1 Stage by stage

| # | Stage | 🟢 Main observation | Decision and limit |
|---|---|---|---|
| 1 | [Early perturbation pilot](../EXPERIMENT_REPORT.md) | All arms answered; so could a question-blind formatting rule | Shortcut invalidated the retrieval interpretation; no matched damaging doses |
| 2 | [Position scorer validation](../scorer/VALIDATION_REPORT.md) | Same 96 target IDs under nested 1K/4K/12K histories on 4 books; identities passed | Go for measurement. **Not** evidence for accumulation |
| 3 | [Dose calibration](../dose_calibration/DOSE_CALIBRATION_REPORT.md) | Alpha and MLP had similar mean short-history excess NLL on calibration books | Doses chosen; the match did not generalise |
| 4 | [Held-out natural text](../heldout_nll/HELDOUT_REPORT.md) | Interaction ≈ −0.0005 nat/token; 4-book interval crossed +0.005 | No support; +0.005 **not excluded**; dose matching failed |
| 5 | [Fresh-book state import](../fresh_state/FRESH_STATE_REPORT.md) | One-seed alpha−MLP state interaction ≈ −0.00045 nat/token | +0.005 **excluded** in that 6-book assay; dose equivalence not established |
| 6 | [Original-model memory gate](../memory_gate/MEMORY_GATE_REPORT.md) | Longer-built R/S improved the 1–95-token mean with 12K KV fixed | Primary 1–16 window **unresolved**; R/S vs attention not fully separated |
| 7 | [Remote-value state gate](../remote_state_gate/REMOTE_STATE_REPORT.md) | Post-question R/S import moved the margin +0.249 nat (8 registries) | Smaller positive effect; +0.5 **excluded** |
| 8 | [R/S and timing follow-up](../remote_components/COMPONENT_REPORT.md) | Post-question S-only +0.228; **pre-question** combined +0.006 | +0.1 pre-question effect **excluded** in this fixed assay; reinjection plausible, not proven |

## 10.2 Reading the arc

1. **Stages 1–4** tried to see gate damage in *behaviour* and loss. The prompts were invalid, then the doses wouldn't match, and the natural-text interaction came out near zero with a wide interval.
2. **Stages 5–6** switched to *state imports* to isolate the recurrent path. Recurrent state built from longer history does help prediction in the secondary window (🟢), but the alpha-specific effect was small enough to exclude +0.005 in one assay.
3. **Stages 7–8** moved to a clean retrieval task with margins. A small post-question state effect exists (🟢). It sits mostly in S (🟢), and it nearly vanishes when the swap happens before the question (🟢). The leading explanation is attention reinjection (🟡).

## 10.3 The current decision

**Go/no-go: stop before a broad alpha/MLP perturbation matrix.** This is a **scope decision** based on these assays. It is **not** a claim that recurrent memory is never useful or that BF16 gate weights can't matter.

Why stop:

- the registries have already been used for design and mechanism work,
- the post-question signal is small compared with the ~11-nat baseline margin,
- the pre-question import excludes the chosen +0.1 effect.

## 10.4 🟣 Proposed next experiment (not implemented, frozen or run)

**Question:** on **fresh** registry families, does a recurrent state built with a remote fact change the answer when the receiver's *recent suffix and attention information* are held constant? The aim is to stop old KV from retrieving the fact during the question, so any remaining effect points at long-range recurrent retention.

Requirements before any margin is interpreted:

- a technically valid way to align source/destination **positions and attention context**,
- an **own-state identity** check,
- full **A–B–A restoration**,
- a **positive sensitivity control**,
- **equal candidate IDs**,
- a **predeclared effect threshold**.

Caveats:

- "Attention-controlled" is an *intended* comparison, not a proven property of any implementation.
- The hybrid state will itself be off the normal trajectory, so even a positive result needs independent replication.

**Branches:**

- A reproducible pre-question S effect on new registries → design an alpha-vs-control weight comparison around it.
- No such effect → report the assay's detection limit and stop that mechanistic line.
- Either way: don't reuse registries 00–15 as "fresh" confirmation.

---

## Capstone exercises

1. **Explain it to a friend (5 minutes, no jargon).** Why did the team stop the perturbation matrix? Your answer should use the words "before the question", "attention" and "excluded" correctly.
2. **Claim audit.** Rewrite this overclaim into three accurate sentences, one each for measured, inferred and proposed: *"Bonsai 2 doesn't use its recurrent memory for long-range facts; attention does all the work."*
3. **Design critique.** For the proposed attention-controlled gate, name one way the "attention control" could fail silently, and one check that would catch it.
4. **Read the source.** Open [COMPONENT_REPORT.md](../remote_components/COMPONENT_REPORT.md) and tag every sentence of its *Interpretation* paragraph as 🟢/🟡/🟣.

<details><summary>Model answers</summary>

1. Swapping the model's recurrent memory right before the question barely changes its answer; the effect is small enough to rule out the size they'd care about (+0.1 nat). Swapping it after the question matters a bit more, probably because attention looks the fact up while reading the question and writes it into memory then. So there's no strong sign that damaged gates would hurt long-range memory, and a big weight-damage study isn't justified yet.
2. 🟢 "In eight registries, importing counterfactual recurrent state before the question changed the answer margin by only +0.006 nat, excluding a +0.1-nat effect in this assay; after the question the effect was +0.249, mostly via S." 🟡 "This fits attention retrieving the fact from KV during the question and writing it into S, but that write was not observed." 🟣 "An attention-controlled gate on fresh registries would test whether S retains the fact without attention's help."
3. Example failure: the receiver's KV or recent suffix still contains a copy or paraphrase of the fact, so attention can still retrieve it. Check: a target-removal control in the receiver context should drive the margin to near zero when the recurrent state is also uninformative; alternatively, verify token-level that no fact tokens remain in the receiver's attention context.
4. Roughly: "real effect of the imported post-question recurrent state" 🟢; "does not support a meaningful effect of the state before the question" 🟢 (exclusion); "the question can retrieve remote values from KV and write a signature into S… leading explanation" 🟡; "could also reflect overwriting or masking" 🟡 (an alternative); the "future test would need fresh registries…" sentence under Go/no-go is 🟣.

</details>

**Congratulations.** You now have the vocabulary and the habits of mind to follow each new Bonsai 2 report. When the next one lands, read it with three questions in mind: *what exactly was measured, over how many independent units, and what path could still explain it?*
