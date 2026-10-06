# Held-out position-matched natural-text comparison

**Decision: the pre-frozen support gate did not pass.** The held-out 1K perturbation doses were not matched, and the alpha-minus-MLP 12K-versus-1K interaction was near zero and slightly negative. This run gives no evidence for the proposed extra long-history alpha damage at these doses, but its four-book interval is wide enough that it does not rule out the predeclared +0.005 nat/token effect. The dose mismatch limits interpretation of the intended matched comparison.

The [freeze](HELDOUT_FREEZE.json) was written before treatment scoring. It fixes the four scorer-validation books, one 96-ID target span at position 30,000 in each, nested 1,024/4,096/12,288-ID histories, all three alpha and three MLP seeds, and the 512/512 CUDA scorer profile. The primary estimand is the document-and-seed mean of `(alpha_12K − alpha_1K) − (MLP_12K − MLP_1K)` on the same target IDs. A positive value means more additional long-history loss for alpha. The smallest meaningful positive effect was fixed at **+0.005 nat per target token**, approximately the size of the matched mean short-history dose in calibration. The support gate also required held-out 1K alpha and MLP mean excess NLLs each in +0.002 to +0.020 and within 0.003 of each other, plus a positive 95% cluster-bootstrap lower bound.

The original-model rerun reproduced every previously saved baseline token NLL exactly (maximum difference **0**). The six variants then scored all **72 document-history-arm-seed cases**, or 6,912 scored token positions, without errors. The [analysis](analyze.py) rechecked source and token-ID hashes, case identities, model and patch hashes, 96-token output lengths, and the scorer profile. Its [enriched results](results.jsonl) retain every gold-token NLL and timing; the [summary](summary.json) contains all per-seed and per-book contrasts. An independent calculation from raw variant outputs, in which the common baseline cancels, reproduced the primary interaction.

| Excess NLL versus original, nat/token | 1K | 4K | 12K |
|---|---:|---:|---:|
| Alpha-only, three-seed mean | +0.003026 | +0.004087 | +0.003136 |
| MLP control, three-seed mean | +0.007664 | +0.007751 | +0.008279 |

At 1K, alpha minus MLP was **−0.004637 nat/token**, exceeding the frozen 0.003 matching tolerance in magnitude. The calibration documents had matched means of +0.005132 and +0.005377; that match did not generalize to this held-out set. One alpha seed averaged only +0.001320 at 1K, and one MLP seed averaged +0.001023, so the per-seed short dose also varied substantially.

The primary 12K-versus-1K interaction was **−0.000506 nat/token**. Its descriptive 95% document-and-paired-seed bootstrap interval was **[−0.007116, +0.005787]**. The secondary 4K-versus-1K interaction was **+0.000973**. The primary was positive for only one of three seeds and one of four documents:

| Group | Primary interaction, nat/token |
|---|---:|
| Seed 20260929 | +0.003000 |
| Seed 20260930 | −0.002354 |
| Seed 20260931 | −0.002163 |
| Pride and Prejudice | −0.003509 |
| Frankenstein | −0.000514 |
| Sherlock Holmes | +0.005211 |
| Dracula | −0.003210 |

The result is weakly against a large, robust alpha-specific history effect at these exact doses, and inconclusive for the +0.005 threshold because of the wide interval and dose mismatch. The held-out books were seen for baseline scorer validation but not for treatment calibration or dose selection. There is only one target span per book, so the document bootstrap has four independent text clusters. This permanent checkpoint intervention cannot by itself isolate recurrent state; full-attention and downstream recurrent blocks can also mediate effects.

**Next gate:** do not reselect doses on these four books and relabel them held out. A stronger test would broaden dose calibration on additional natural documents, freeze a fresh unseen held-out set with more target spans, and establish out-of-sample 1K dose matching before interpreting the long-history interaction. The retrieval task also remains at exact-accuracy ceiling; answer likelihood must first be validated for both correct and wrong codes, or the task made harder while retaining answer-swap and shortcut audits. No further perturbation matrix was started after this failed gate. The old frozen inputs and unrelated `.gitignore` and `uv.lock` changes were left untouched.
