"""Pure metrics and the predeclared branch rule for the read-out diagnostics. No I/O, no torch."""
from __future__ import annotations

import collections
import math
import re

LETTERS = "ABCD"
_ANSWER = re.compile(r"[Aa]nswer\s*(?:is\s*)?[:：]?[\s*]*[:：]?[\s*(]*([ABCD])(?![A-Za-z0-9])")
_BOXED = re.compile(r"\\boxed\{\s*\(?([ABCD])\)?\s*\}")


def wilson(k: int, n: int) -> dict:
    """Identical formula and keys to prepare_qat08_chat.wilson (percent)."""
    z = 1.96
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / den
    return {"n": n, "correct": k, "mean": 100*p, "lo": 100*(mid-half), "hi": 100*(mid+half)}


def mean_ci(values: list[float]) -> dict:
    """Normal 95% interval over independent items; values in 0..1, result in percent."""
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {"mean": 100*mu, "lo": 100*(mu-half), "hi": 100*(mu+half), "n": n}


def renorm(values: list[float]) -> list[float]:
    top = max(values)
    z = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - z for v in values]


def item_pairwise(scores: list[float], gold: int) -> float:
    others = [k for k in range(len(scores)) if k != gold]
    return sum(1.0 if scores[gold] > scores[k] else 0.5 if scores[gold] == scores[k] else 0.0
               for k in others) / len(others)


def _gold(case_id: str) -> int:
    return LETTERS.index(case_id.rsplit("/", 1)[1])


def _strict_hit(values: list[float], gold: int) -> float:
    return float(values[gold] > max(v for k, v in enumerate(values) if k != gold))


def letter_metrics(rows: dict[str, list[float]]) -> dict:
    """Letter-logit metrics. Calibration subtracts the arm's mean renormalized log-probability per letter."""
    ids = list(rows)
    gold = [_gold(i) for i in ids]
    raw = [rows[i] for i in ids]
    norm = [renorm(v) for v in raw]
    bias = [sum(w[k] for w in norm) / len(norm) for k in range(4)]
    cal = [[w[k] - bias[k] for k in range(4)] for w in norm]
    raw_hits = [_strict_hit(v, g) for v, g in zip(raw, gold)]
    cal_hits = [_strict_hit(c, g) for c, g in zip(cal, gold)]
    pw = [item_pairwise(c, g) for c, g in zip(cal, gold)]
    ci = mean_ci(pw)
    predicted = collections.Counter(LETTERS[max(range(4), key=lambda k: v[k])] for v in raw)
    return {"items": len(ids), "raw_accuracy": 100*sum(raw_hits)/len(ids),
            "calibrated_accuracy": 100*sum(cal_hits)/len(ids), "calibrated_hits": cal_hits,
            "pairwise": ci["mean"], "pairwise_lo": ci["lo"], "pairwise_hi": ci["hi"], "pairwise_items": pw,
            "letter_mass": sum(sum(math.exp(x) for x in v) for v in raw) / len(raw),
            "bias": bias, "predicted": dict(predicted)}


def cloze_metrics(rows: dict[str, list[float]]) -> dict:
    """Option-likelihood metrics. Case ids are subject/index/GOLD/k; score = -mean token NLL of option k."""
    items: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for case_id, nll in rows.items():
        key, k = case_id.rsplit("/", 1)
        items[key][int(k)] = -sum(nll) / len(nll)
    pw, hits = [], []
    for key, options in items.items():
        assert sorted(options) == [0, 1, 2, 3], key
        scores = [options[k] for k in range(4)]
        g = _gold(key)
        pw.append(item_pairwise(scores, g))
        hits.append(_strict_hit(scores, g))
    ci = mean_ci(pw)
    return {"items": len(pw), "accuracy": 100*sum(hits)/len(hits), "pairwise": ci["mean"],
            "pairwise_lo": ci["lo"], "pairwise_hi": ci["hi"], "pairwise_items": pw}


def binding_metrics(rows: dict[str, list[float]]) -> dict:
    r = letter_metrics(rows)
    n = r["items"]
    raw_k = round(r["raw_accuracy"] * n / 100)
    cal_k = round(sum(r["calibrated_hits"]))
    return {**r, "raw_wilson": wilson(raw_k, n), "calibrated_wilson": wilson(cal_k, n)}


def summary(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in ("pairwise_items", "calibrated_hits")}


def parse_mc_answer(text: str) -> str | None:
    """Last 'Answer: X' (markdown bold/parentheses allowed) or \\boxed{X}; 'answer is A <word>' is the article."""
    found = [(m.start(), m.group(1)) for m in _BOXED.finditer(text)]
    for m in _ANSWER.finditer(text):
        if m.group(1) == "A" and re.match(r"\s+[a-z]", text[m.end():]):
            continue
        found.append((m.start(), m.group(1)))
    return max(found)[1] if found else None


def regions(n_tokens: int, prefix_len: int, suffix_len: int) -> dict[str, tuple[int, int]]:
    """Logit positions (half-open) whose prediction targets the question, the template suffix, or the answer.

    Position t predicts token t+1; position n-1 predicts the first answer token.
    """
    n = n_tokens
    return {"question": (prefix_len - 1, n - suffix_len - 1), "template": (n - suffix_len - 1, n - 1),
            "letter": (n - 1, n)}


def branch(results: dict, rule: dict) -> dict:
    """Predeclared mapping from diagnostic outcomes to the next experiment's arms (review section 6)."""
    fp = results["arms"]["fp"]
    chat = results["arms"]["qat_chat_42-final"]
    reasons = []
    if fp["binding"]["calibrated_accuracy"] < rule["fp_binding_min"] or \
            fp["letter"]["pairwise"] < rule["fp_letter_pairwise_min"]:
        return {"branch": "harness_invalid", "arms_required": [], "arms_optional": [], "stop": True,
                "reasons": ["FP failed the harness sanity floor; investigate before any experiment"]}
    bind = chat["binding"]["calibrated_accuracy"]
    knowledge = max(chat["cloze_chat"]["pairwise_lo"], chat["cloze_raw"]["pairwise_lo"]) > rule["knowledge_pairwise_lo_above"]
    cloze_informative = max(fp["cloze_chat"]["pairwise_lo"], fp["cloze_raw"]["pairwise_lo"]) > rule["fp_cloze_pairwise_lo_min"]
    traj = results["trajectory"]["qat_chat_42"]["final_minus_t8m"]
    rising = any(traj[k]["mean"] >= rule["trajectory_rise_points"] and traj[k]["lo"] > 0 for k in ("letter", "cloze_raw"))
    reasons.append(f"chat binding calibrated accuracy {bind:.1f}%")
    reasons.append(f"cloze knowledge signal: {knowledge}")
    reasons.append(f"rising trajectory: {rising}")
    reasons.append(f"FP cloze informative (lower bound > {rule['fp_cloze_pairwise_lo_min']}): {cloze_informative}")
    if bind < rule["readout_broken_below"]:
        name, required = "readout_broken", ["M", "U"]
    elif bind >= rule["readout_intact_at_least"] and not cloze_informative:
        name, required = "ambiguous", ["M", "U"]
    elif bind >= rule["readout_intact_at_least"] and not knowledge:
        name, required = "knowledge_lost", ["U", "B"]
    elif bind >= rule["readout_intact_at_least"]:
        name, required = "closed_book_readout", ["M", "U"]
    else:
        name, required = "ambiguous", ["M", "U"]
    if rising and "B" not in required:
        required = required + ["B"]
    optional = ["MU"] if "M" in required else []
    return {"branch": name, "arms_required": required, "arms_optional": optional, "stop": False, "reasons": reasons}
