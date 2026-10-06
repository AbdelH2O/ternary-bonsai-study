# Corrected retrieval prompt preparation (CPU only)

The new [frozen prompts](prompts/retrieval_v2.jsonl) contain 16 independent
registries with 16 candidate records each. Two different records are selected
for questions **after** all names and codes are generated. Every name has the
same three-part format, every code has the same `K-` plus six-digit format,
and codes are sampled independently of names, positions, and questions.
Within each registry, a counterfactual swaps the two queried records' codes;
both gold answers follow the swap while the question text stays unchanged.

Each question/swap appears in three layouts. `short-near` has six neutral
background paragraphs before the registry. `long-near` has 200 before it;
`long-far` moves those *same* 200 paragraphs after the unchanged registry.
All layouts retain 16 candidate records. The actual GGUF Jinja chat template
is rendered as one user message with an assistant generation prompt and
thinking disabled, then counted by the pinned Prism `llama-tokenize --no-bos`.

| Layout | Chat tokens | Tokens from target-record end to question |
| --- | ---: | ---: |
| Short near | 1,036–1,044 | 0–594 |
| Long near | 12,281–12,310 | 0–594 |
| Long far | 12,281–12,310 | 11,591–12,192 |

Every long-near/long-far pair has **exactly equal total chat-token count** and
the same registry, answer, and filler text. The distance measure subtracts
GGUF-tokenized prefixes at the target-record end and question start; a token
at a cut boundary may change the count by roughly one.

The [machine audit](prompts/retrieval_v2_audit.json) checks 32 independent
base questions, plus their other layouts and value swaps. A fixed first-record
answer gets 1/32; last-record gets 3/32. The best of all 16 fixed record
positions gets 4/32, versus 2/32 expected for a prespecified position at
1/16 chance. Lexicographically first name gets 3/32; first code 0/32;
smallest name suffix 1/32. There is no unique code prefix or name format.
Across the 32 base questions, all gold codes are distinct and none is the
target name's numeric suffix. These simple question-blind, context-blind,
position, prefix, and formatting checks found no deterministic shortcut.
The selected best-of-16 fixed-position rate is exploratory and inflated by
selection over 16 rules; it is not an accuracy estimate for a model.

Reproduce from the repository root with
`python3 analysis/bonsai2/build_retrieval_v2.py` and
`python3 analysis/bonsai2/audit_retrieval_v2.py` in a clean copy without the
frozen output files. The builder refuses to overwrite frozen prompts, while
the auditor verifies an existing audit without modifying it. The
[manifest](prompts/retrieval_v2_manifest.json) records the seed, source model
hash (verified against the local GGUF), tokenizer binary hash, chat-render
fixture hash, each prompt/chat hash, and the JSONL SHA-256. The generation
script leaves all earlier frozen inputs and `uv.lock` untouched.

This preparation runs **no Bonsai inference**. Baseline accuracy, response
sensitivity to question/value swaps, and perturbation effects remain untested.
These prompts are ready for a small retrieval pilot only after the scorer and
perturbation-dose gates in [NEXT_EXPERIMENT_DESIGN.md](NEXT_EXPERIMENT_DESIGN.md).
