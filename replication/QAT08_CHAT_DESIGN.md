# Chat-aware 0.8B amendment — frozen design, 2026-10-02

**Decision pending:** the design is frozen; full teacher generation and student training await the user's go-ahead. No new capability result exists yet.

The immutable [design](results/qat08_chat/design.json), SHA-256 `185631c73fac6de20243538401899471bf8ac9ce8fe4993656f927f02b5ec0fd`, fixes prompts, source/model/code hashes, evaluation cases, generation and packing rules, recipe, and decisions. The [freeze script](freeze_qat08_chat.py) has two stages. After full teacher generation, its `protocol` stage adds exact generated-data hashes and the complete contamination audit to `results/qat08_chat/protocol.json`, before any student training. It cannot change the approved design's decisions or overwrite either freeze.

Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Data and training

| Item | Frozen choice |
|---|---|
| Instruction source | [NVIDIA Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater/tree/ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40), revision `ae79f8ac`; synthetic subsets only |
| License | The pinned dataset card licenses these subsets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), permitting commercial adaptation with attribution. Credit: NVIDIA / Wang et al., *HelpSteer2* (2024). Original answers are discarded. |
| Prompts | 10,000 first user turns: 6,400 general, 1,500 math, 1,000 precise instruction following, 1,000 complex instructions, 100 JSON-format instructions; seeded selection, 24–384 template tokens |
| Responses | The original FP F32 GGUF in the pinned llama-server, Qwen3.5 template, thinking disabled, greedy, 8 slots, up to 768 new tokens |
| Chat filtering | Keep nonempty, naturally terminated responses without special/thinking tokens or benchmark-content 13-word overlaps; complete transcript ≤1,024 tokens; require ≥3,000 retained chats and ≥1M unique chat tokens or stop before training |
| Mix | Exactly 25% chat / 75% FineWeb tokens: 16.384M / 49.152M. Every microbatch has three sequential FineWeb sequences and one chat sequence. Chat transcripts are concatenated and cycled without padding; final report must disclose unique tokens and repetition count. |
| Student | One **42-zero** arm, initialized from the same folded FP weights as QAT08 |
| Unchanged recipe | H512; 65.536M tokens; 2,000 steps; microbatch 4 × accumulation 8 × 1,024 tokens; peak LR 1e-4; identical warmup/cosine schedule, AdamW BF16 moments, all-position forward KL, frozen embedding and non-projection tensors |

🟡 A quarter of the tokens gives chat delimiters, answer starts and endings 16.4M tokens of exposure while preserving three quarters of the prose budget. This tests the chat-data hypothesis; it does not determine an optimal mixture. Repeating a smaller synthetic corpus is a limitation. There is no second training arm because QAT08's 42-zero rule matched or beat the free rule, and one arm fits the compute budget.

## Evaluation and decisions

The QAT08 harness is retained: 1,024-token book slices with targets 64–1023, 5,330 MMLU-Redux letter-logit items, 1,319 greedy GSM8K items with the same boxed/last-number parser and 1,024-token generation limit, and 200 raw-text retrieval registries. All arms see identical IDs within a comparison. FP, GPTQ and both QAT08 trained arms are comparators.

🟢 Fifteen fresh books have verified downloaded Title headers and Gutenberg IDs absent from earlier freeze dictionaries: three validation books (*Bleak House*, *The Return of the Native*, *The Mayor of Casterbridge*) and twelve held out. All comparators will be rescored on these books. The unchanged task/retrieval case hashes permit reuse of hash-checked QAT08 comparator outputs; book results will not be pooled across experiments.

🟢 The 10,000 prompts passed a literal contamination audit against every frozen MMLU/GSM8K case: no shared normalized 13-word content gram, full-question equality, or embedded 4–12-word question. This does not detect paraphrases. Before training, every evaluation case will also receive QAT08's rolling **13-token** check against the entire actual mixed stream. Any smaller overlaps will be reported; a majority-overlap case blocks the protocol freeze and training.

| Gate | Predeclared requirement |
|---|---|
| MMLU recovered | Two-sided 95% Wilson lower accuracy bound **>30%**, clearly above 25% chance |
| GSM8K recovered | Wilson lower bound **>10%**, also >10% when every output without natural EOS is counted wrong |
| Chat recovered | Both task gates pass |
| Allowed book cost | Upper two-sided 95% paired t bound over 12 books for new arm minus QAT08's 42-zero arm **≤+0.10 nat/token** |
| Amendment success | Chat recovery and allowed book cost both pass |

Retrieval margins/accuracy, answer-letter distribution, letter probability mass, termination rates and the checkpoint book-loss curve are diagnostics. The final checkpoint is fixed; validation cannot select another checkpoint or change thresholds.

🟢 The negative control is QAT08's degenerate GSM8K outputs. Accidental parser accuracy was 1/1,319 for GPTQ, 14/1,319 (1.061%) for the free arm and 3/1,319 for the 42-zero arm. The largest Wilson upper bound is **1.774%**; randomly reassigning gold answers gives **1.258%** expected for the free arm. A lower bound above 10%, including a natural-termination gate, is well clear of this baseline. It establishes substantial recovery, not FP equivalence.

## Cost and execution

🟢 The 64-prompt teacher pilot took **58.5 seconds**, generating 34,975 tokens at **598 tokens/s**. Thirty responses terminated naturally; 34 hit the limit and will be excluded. All 30 completed transcripts passed exact token-ID equality against the non-thinking chat template. The unchanged GPU quantizer self-test passed rotation, packing, 42-zero counts and STE gradients.

🟡 Full teacher generation projects to **2.6 GPU-hours**. Student training uses QAT08's measured throughput: about **6 hours**, with a 6.5-hour ceiling. Allow up to **2 hours** for export/evaluation. Total planned cost is approximately **10–11 GPU-hours**, with a 12-hour ceiling. These are estimates; actual time will be reported.

Everything uses `work/qat08_chat` and `results/qat08_chat`. Long GPU jobs run as systemd user units. Teacher outputs resume by response; training uses the original atomic `resume.pt`; evaluation resumes by scorer shard and GSM8K item. The new training entrypoint requires an explicit user-approval record matching both design and protocol hashes. No commits are authorized.

## Preparation incidents

- Title checks caught wrong remembered IDs before case construction: 668 was a dictionary and 427 was *The Great War Syndicate*. Those downloads are preserved as rejected artifacts. Correct IDs are 6688 and 4274. A separate title check corrected Villette from 918 to 9182 before its experiment download.
- Only 135 JSON-format prompts met the length cap. The draft quota was reduced from 500 to 100, and the general quota increased from 6,000 to 6,400, before generation or evaluation.
- The installed tokenizer returns an encoding object rather than a plain ID list for `tokenize=True`. The packing assertion now renders text then encodes it, matching the existing harness; the completed pilot transcripts pass.

## Reproduction and pending work

With the project venv, from `analysis/bonsai2/replication`:

```bash
python prepare_qat08_chat.py cases       # done; title-checked fresh books, unchanged task cases
HF_HUB_DISABLE_XET=1 python prepare_qat08_chat.py prompts  # done
python prepare_qat08_chat.py pilot       # done as systemd user unit on unsandboxed GPU
python prepare_qat08_chat.py audit       # done
python qat_08b_chat.py selftest          # done on unsandboxed GPU
python freeze_qat08_chat.py design       # done; refuses to overwrite
python freeze_qat08_chat.py verify-design

# Only after the user's go-ahead, as unsandboxed systemd user units:
# /bin/bash run_qat08_chat_prepare.sh    # teacher generation, pack, protocol freeze, verify
# Record the user's approval bound to the actual design/protocol hashes.
# /bin/bash run_qat08_chat.sh            # unchanged, resumable student training
# python score_qat08_chat.py export
# python score_qat08_chat.py curve
# python score_qat08_chat.py heldout
# python score_qat08_chat.py gsm8k
# python score_qat08_chat.py analyze
```

After scoring, write `QAT08_CHAT.md` with the frozen decision first, comparison tables, evidence tags, actual costs, limitations, incidents and reproduction commands. Then update the plan status and add the requested `Update:` link in `QAT08.md`. Those result updates await actual results.
