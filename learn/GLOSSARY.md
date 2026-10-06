# Glossary

One line per term. The number in brackets is the lesson that explains it. [Course home](README.md)

## Model and representation

| Term | Meaning |
|---|---|
| **Token** | Integer ID for a piece of text, often a word fragment [1] |
| **Language model** | Assigns probabilities to the next token given the previous ones [1] |
| **Weights / checkpoint** | Saved numerical parameters; persist across prompts [1] |
| **Runtime state** | Temporary memory built while reading one prompt (KV, R, S) [1] |
| **Original / baseline** | The downloaded Bonsai 2 checkpoint (not full-precision Qwen) [1] |
| **Variant** | A copy of the original with a recorded weight patch [1] |
| **Prompt / prefix / continuation** | Input text / part before a boundary / text after it [1] |
| **GGUF** | File format holding weights, tensor metadata and tokenizer [1] |
| **Quantization** | Storing weights with fewer bits; here `PQ2_0` ternary groups [1] |
| **F32 / F16 / BF16** | 32-bit float / 16-bit precise-but-narrow / 16-bit wide-but-coarse [1] |
| **Hadamard transform** | Structured orthogonal rotation; stored matrices live in this rotated basis [1] |
| **Block** | One layer transforming each token's vector; 64 in total [1] |
| **Head** | Group of channels with its own attention/recurrent parameters [1] |
| **MLP** | Multilayer perceptron inside each block; `ffn_up` used as a control [1, 6] |

## Information paths

| Term | Meaning |
|---|---|
| **Full attention** | Block that can read every earlier token's stored keys/values; 16 of them [1] |
| **KV cache** | Stored keys/values for full attention; grows with the prompt (F16 here) [1] |
| **Recurrent / linear-attention block** | Keeps a fixed-size state updated per token; 48 of them [1] |
| **Gated DeltaNet (GDN)** | The recurrent computation in the 48 linear blocks [2] |
| **R** | Short convolution buffer of recent inputs in each recurrent block (F32) [1] |
| **S** | Gated DeltaNet recurrent state; compressed summary (F32) [1] |
| **`ssm_alpha`** | BF16 weights feeding the decay/retention gate [2] |
| **`ssm_beta`** | BF16 weights feeding the write strength via sigmoid [2] |
| **`ssm_a`, `ssm_dt.bias`** | Small F32 tensors in the decay formula `softplus(alpha+dt_bias)·ssm_a` [2] |
| **Gate** | Input-dependent control of forget/retain/update [2] |
| **Decay horizon** | 1/\|g\| for a fixed zero-input decay; not a measured retention interval [2] |
| **Interference** | Later inputs altering or competing with earlier useful information [2] |
| **Attention reinjection** | 🟡 Inferred path: attention reads the old record at question time, then S gets a trace [7] |

## Measurement

| Term | Meaning |
|---|---|
| **Logit** | Unnormalised score per vocabulary token [3] |
| **Softmax** | Converts logits to probabilities summing to 1 [3] |
| **NLL** | `−ln q` for the true token's probability q, in nats; lower is better [3] |
| **Nat** | Unit of log-probability using natural log [3] |
| **Mean NLL** | Average NLL over a *specified* set of targets (the denominator) [3] |
| **Perplexity** | `e^(mean NLL)`; the same information in another form [3] |
| **Excess NLL** | Variant mean NLL − original mean NLL on the same targets [3, 4] |
| **Teacher forcing** | Feed true previous tokens; score each next true token [3] |
| **Tokenizer / chat template** | Text→IDs / role markers and formatting around user text [3] |
| **BOS / end-of-turn** | Beginning-of-sequence token / token closing the scored answer [3] |
| **Context window / history length** | Configured capacity / IDs actually supplied [3] |
| **Prefill / decode** | Bulk prompt processing / step-by-step evaluation [3] |
| **Position matching** | Same target IDs at the same absolute positions across conditions [4] |
| **Nested suffixes** | Histories that all end at the same token, each the tail of the next [4] |
| **Sparse scorer** | Records only gold-token NLLs, not full logits (sparse *output*) [4] |
| **Retrieval task** | Ask for a fact placed in the prompt [5] |
| **Registry** | One independently generated set of item→code records [5] |
| **Target / distractor / filler** | Queried record / other plausible records / intervening text [5] |
| **short-near / long-near / long-far** | ~1K near / ~12K near / ~12K with target far from question [5] |
| **Shortcut / shortcut audit** | Question-blind rule that answers anyway / check that none works [5] |
| **Exchangeable** | Target and distractor formats indistinguishable [5] |
| **Value swap** | Exchange two remote codes; the correct answer must flip [5] |
| **Target removal control** | Delete the queried record; margin should drop [5] |
| **Exact accuracy / ceiling** | Share exactly correct / all arms perfect, so differences are hidden [5] |
| **Gold / wrong answer** | Correct complete answer / plausible incorrect one [5] |
| **Answer margin M** | `NLL(wrong) − NLL(gold)`, in nat per answer [5] |
| **Own-minus-counterfactual effect** | Margin with own state − margin after counterfactual import [5, 7] |
| **Answer F1** | Harmonic mean of token precision and recall after normalisation [5] |
| **Temperature 0** | Always choose the highest-scoring token [5] |

## Experimental design

| Term | Meaning |
|---|---|
| **Perturbation** | Deliberate numerical change to test sensitivity [6] |
| **Reflink** | Copy-on-write file copy; source stays read-only [6] |
| **Patch manifest** | Record of exactly which weights/scales changed and by how much [6] |
| **Seed** | Makes random sign patterns reproducible [6] |
| **Coverage / strength** | Fraction of groups selected / scale change applied [6] |
| **Dose** | Observed short-history excess NLL, not nominal scale [6] |
| **Dose matching** | Arms cause similar short-history damage before comparing long-history effects [6] |
| **Calibration / held-out / fresh data** | Tune on / evaluate frozen choices on / never used for any prior choice [6] |
| **Baseline rerun** | Re-score the original to measure reproducibility [6] |
| **Control arm / positive control** | Comparison for nonspecific damage / a change the assay must detect [6] |
| **Interaction (difference of differences)** | `[α12K−α1K] − [MLP12K−MLP1K]` [6] |

## State methods

| Term | Meaning |
|---|---|
| **State capture** | Save runtime memory after a known prefix [7] |
| **Full-state / recurrent-only (`PARTIAL_ONLY`)** | KV+R+S / R+S only, leaving receiver KV [7] |
| **Donor / receiver (destination)** | Produced the saved state / continues scoring after import [7] |
| **Own-state / counterfactual import** | Same prompt's R/S / paired swap prompt's R/S [7] |
| **A–B–A restoration** | Direct, import, restore, direct again; the two A's must match [7] |
| **Roundtrip** | Save then reload the same state; score must match [7] |
| **Unrelated-book positive control** | Import a very different state; score must change [7] |
| **Cell position metadata / rebase** | Positions stored in state files / rewriting that 4-byte field [7] |
| **Target 0 exclusion** | Drop the first target when it was scored before the import [7] |
| **Off-trajectory / hybrid state** | A mix the model never produces normally [7] |
| **Reciprocal imports** | Swap donor/receiver both ways; not an additive decomposition [7] |
| **Readout** | Receiver computations turning state + input into logits [7] |
| **Component splice** | Replace only R or only S payload bytes in a state file [7] |
| **R/S interaction** | Combined effect − (R-only + S-only) [7] |
| **Layer-selective splice** | 🟣 Proposed: S from early vs post-attention blocks [7] |

## Reproducibility

| Term | Meaning |
|---|---|
| **Pin** | Use one exact software release [8] |
| **SHA-256 hash** | Byte-level fingerprint of a file [8] |
| **Freeze / preregistration** | Fix inputs, estimands, thresholds, code before seeing treatment results [8] |
| **Technical amendment** | Versioned implementation fix preserving the failed version [8] |
| **JSONL** | One JSON result per line [8] |
| **Artifact inventory** | Hash list of final outputs [8] |
| **Logical batch / physical microbatch** | Tokens submitted together / internal execution unit [8] |
| **Numerical floor** | Score variation from execution details alone [8] |
| **Host-device CUDA** | Giving the binary GPU access; execution requirement, not a treatment [8] |

## Inference

| Term | Meaning |
|---|---|
| **Assay** | One fully specified measurement procedure [9] |
| **Estimand** | The exact quantity being estimated [9] |
| **Smallest meaningful effect** | Threshold fixed before results [9] |
| **Gate / support / exclusion / unresolved** | Predeclared rule / prediction supported / effect ≥ T ruled out / neither [9] |
| **Go/no-go** | Whether evidence justifies the next experiment [9] |
| **Confidence interval** | Range from a procedure with stated long-run coverage [9] |
| **Cluster bootstrap** | Resample whole books/registries (and seeds) [9] |
| **t interval** | `mean ± t·sd/√n` over cluster means [9] |
| **Independent unit (cluster)** | The book or registry, not the token [9] |
| **Power** | Chance a gate detects a specified true effect [9] |
| **Equivalence** | Whole interval inside a tolerance band [9] |
| **Sensitivity analysis** | Leave-one-out and alternative checks that qualify the primary result [9] |
| **Multiple looks / post hoc** | Repeated examination / chosen after seeing results [9] |
| **Replication** | Same question and analysis on genuinely new samples [9] |
| **Prediction / observation / mechanism** | Hypothesis / valid measurement / controlled causal account [9] |
| **Measured / inferred / proposed** | 🟢 completed run / 🟡 compatible explanation / 🟣 future assay [README] |
