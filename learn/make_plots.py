"""Generate the data-driven SVG figures for the Bonsai 2 learning set.

Pure standard library, no plotting dependency. Run from any directory:
    python3 analysis/bonsai2/learn/make_plots.py
Measured values are copied from the linked reports; toy values are labelled as such.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "img"
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
INK, MUTED, GRID = "#1a202c", "#4a5568", "#e2e8f0"
GREEN, AMBER, PURPLE, RED = "#2f855a", "#b7791f", "#6b46c1", "#c53030"
BLUE, ORANGE, TEAL = "#3b5b92", "#c05621", "#2c7a7b"


def svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n'
    )


def text(x, y, s, size=13, fill=INK, anchor="start", weight="400"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


def line(x1, y1, x2, y2, stroke=MUTED, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def polyline(points, stroke, width=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}"{d}/>'


def rect(x, y, w, h, fill, stroke="none", rx=0):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def circle(x, y, r, fill, stroke="none"):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def decay_horizon():
    """Zero-input decay e^{g t} for a few fixed g; marks t = 1/|g|."""
    w, h = 820, 420
    x0, y0, pw, ph = 80, 60, 560, 270
    tmax = 400
    gs = [(-0.1, ORANGE), (-0.02, AMBER), (-0.005, BLUE)]
    X = lambda t: x0 + pw * t / tmax
    Y = lambda v: y0 + ph * (1 - v)
    b = [text(w / 2, 30, "Toy: a fixed decay rate g shrinks an untouched state by e^g per step", 18, anchor="middle", weight="700")]
    for v in (0, 0.25, 0.5, 0.75, 1):
        b.append(line(x0, Y(v), x0 + pw, Y(v), GRID))
        b.append(text(x0 - 8, Y(v) + 4, f"{v:.2f}", 12, MUTED, "end"))
    for t in range(0, tmax + 1, 100):
        b.append(text(X(t), y0 + ph + 20, str(t), 12, MUTED, "middle"))
    b.append(text(x0 + pw / 2, y0 + ph + 42, "steps (tokens) with NO new input", 13, MUTED, "middle"))
    b.append(text(x0 - 8, y0 - 14, "fraction remaining", 13, MUTED, "start"))
    b.append(line(x0, Y(1 / math.e), x0 + pw, Y(1 / math.e), RED, 1, "4 4"))
    b.append(text(x0 + pw - 4, Y(1 / math.e) - 6, "1/e ≈ 0.37", 12, RED, "end"))
    for g, col in gs:
        pts = [(X(t), Y(math.exp(g * t))) for t in range(0, tmax + 1, 2)]
        b.append(polyline(pts, col))
        horizon = 1 / abs(g)
        if horizon <= tmax:
            b.append(circle(X(horizon), Y(1 / math.e), 5, col))
    ly = 90
    for g, col in gs:
        b.append(line(665, ly, 695, ly, col, 3))
        b.append(text(703, ly + 4, f"g = {g}", 13))
        b.append(text(703, ly + 21, f"horizon 1/|g| = {1 / abs(g):.0f}", 12, MUTED))
        ly += 55
    b.append(text(w / 2, 404, "The real gate depends on the input and S keeps receiving writes: a horizon is NOT a measured retention interval.", 12.5, RED, "middle", "600"))
    return svg(w, h, "\n".join(b))


def softmax_nll():
    """Toy logits -> softmax probabilities -> NLL of the true token."""
    toks = [" mat", " floor", " sofa", " moon"]
    logits = [3.0, 2.0, 1.0, -1.0]
    z = sum(math.exp(l) for l in logits)
    probs = [math.exp(l) / z for l in logits]
    w, h = 860, 380
    b = [text(w / 2, 30, 'Toy: next token after "The cat sat on the"', 18, anchor="middle", weight="700")]
    # logits panel
    b.append(text(150, 64, "1. logits (any real number)", 14, BLUE, "middle", "700"))
    base = 250
    for i, (t, l) in enumerate(zip(toks, logits)):
        x = 50 + i * 52
        hgt = l * 45
        y = base - hgt if l >= 0 else base
        b.append(rect(x, y, 40, abs(hgt), "#e6edf8", BLUE))
        b.append(text(x + 20, (y - 6) if l >= 0 else (y + abs(hgt) + 16), f"{l:+.1f}", 12, INK, "middle"))
        b.append(text(x + 20, 335, t.strip(), 12, MUTED, "middle"))
    b.append(line(40, base, 262, base, MUTED))
    # arrow
    b.append(text(310, 190, "softmax", 14, MUTED, "middle", "700"))
    b.append(text(310, 208, "p = eˡ / Σ eˡ", 12, MUTED, "middle"))
    b.append(line(280, 220, 340, 220, MUTED, 2))
    b.append('<path d="M340,215 L350,220 L340,225 z" fill="#4a5568"/>')
    # probability panel
    b.append(text(485, 64, "2. probabilities (sum to 1)", 14, GREEN, "middle", "700"))
    for i, (t, p) in enumerate(zip(toks, probs)):
        x = 380 + i * 52
        hgt = p * 220
        b.append(rect(x, base - hgt, 40, hgt, "#c6f6d5" if i else "#68d391", GREEN))
        b.append(text(x + 20, base - hgt - 6, f"{p:.3f}", 12, INK, "middle"))
        b.append(text(x + 20, 335, t.strip(), 12, MUTED, "middle"))
    b.append(line(370, base, 592, base, MUTED))
    # nll panel
    q = probs[0]
    b.append(text(730, 64, "3. loss for the TRUE token", 14, RED, "middle", "700"))
    b.append(rect(630, 90, 205, 170, "#fff5f5", RED, 10))
    b.append(text(732, 120, 'true next token: "mat"', 13, INK, "middle"))
    b.append(text(732, 148, f"q = {q:.3f}", 15, INK, "middle", "700"))
    b.append(text(732, 178, f"NLL = −ln q = {-math.log(q):.3f} nat", 15, RED, "middle", "700"))
    b.append(text(732, 208, 'if the truth were "moon":', 12, MUTED, "middle"))
    b.append(text(732, 228, f"NLL = −ln {probs[3]:.3f} = {-math.log(probs[3]):.2f} nat", 12, MUTED, "middle"))
    b.append(text(732, 250, "lower is better", 12, MUTED, "middle"))
    b.append(text(w / 2, 360, "Perplexity = e^(mean NLL). It re-expresses the same average; it is not extra evidence.", 13, INK, "middle", "600"))
    return svg(w, h, "\n".join(b))


def nll_curve():
    w, h = 700, 330
    x0, y0, pw, ph = 70, 50, 560, 220
    X = lambda q: x0 + pw * q
    Y = lambda v: y0 + ph * (1 - v / 5)
    b = [text(w / 2, 28, "NLL = −ln q : confident-and-right is cheap, confident-and-wrong is expensive", 15, anchor="middle", weight="700")]
    for v in range(0, 6):
        b.append(line(x0, Y(v), x0 + pw, Y(v), GRID))
        b.append(text(x0 - 8, Y(v) + 4, str(v), 12, MUTED, "end"))
    for q in (0, 0.25, 0.5, 0.75, 1):
        b.append(text(X(q), y0 + ph + 18, f"{q:g}", 12, MUTED, "middle"))
    b.append(text(x0 + pw / 2, y0 + ph + 40, "q = probability the model gave the correct token", 13, MUTED, "middle"))
    b.append(text(22, y0 + ph / 2, "nats", 13, MUTED, "middle"))
    pts = [(X(q), Y(-math.log(q))) for q in [i / 500 for i in range(4, 501)]]
    b.append(polyline(pts, RED))
    for q in (0.1, 0.5, 0.9):
        v = -math.log(q)
        b.append(circle(X(q), Y(v), 5, RED))
        b.append(text(X(q) + 9, Y(v) - 8, f"q={q}: {v:.3f}", 12, INK))
    return svg(w, h, "\n".join(b))


def gates():
    """Schematic of support / exclusion / unresolved plus the measured intervals."""
    w, h = 900, 590
    b = [text(w / 2, 30, "Reading a gate: where does the interval sit relative to 0 and the threshold?", 18, anchor="middle", weight="700")]
    # schematic
    x0, pw = 250, 520
    lo, hi = -0.4, 1.0
    X = lambda v: x0 + pw * (v - lo) / (hi - lo)
    thr = 0.5
    top = 60
    b.append(text(40, top + 10, "SCHEMATIC (threshold T)", 13, MUTED, weight="700"))
    b.append(line(X(0), top + 20, X(0), top + 170, MUTED, 1.2))
    b.append(text(X(0), top + 185, "0", 12, MUTED, "middle"))
    b.append(line(X(thr), top + 20, X(thr), top + 170, PURPLE, 1.5, "5 4"))
    b.append(text(X(thr), top + 185, "T (smallest meaningful effect)", 12, PURPLE, "middle"))
    cases = [
        ("Support", "whole interval above 0, estimate ≥ T", 0.72, 0.55, 0.9, GREEN),
        ("Exclusion", "upper bound below T", 0.18, 0.02, 0.36, AMBER),
        ("Unresolved", "neither rule passes (e.g. spans 0 to T)", 0.2, -0.25, 0.7, MUTED),
    ]
    for i, (name, rule, est, a, c, col) in enumerate(cases):
        y = top + 45 + i * 45
        b.append(text(40, y + 4, name, 14, col, weight="700"))
        b.append(text(40, y + 20, rule, 11.5, MUTED))
        b.append(line(X(a), y, X(c), y, col, 4))
        b.append(circle(X(est), y, 6, "#fff", col))
    # measured
    top2 = 290
    b.append(text(40, top2, "MEASURED (each row has its own units and threshold — do not compare rows)", 13, MUTED, weight="700"))
    rows = [
        ("Held-out natural text", "interaction, nat/token", -0.000506, -0.007116, 0.005787, 0.005, "unresolved", MUTED, 0.012),
        ("Memory gate, targets 1–16", "memory benefit, nat/token", 0.080682, -0.013169, 0.174534, 0.005, "unresolved", MUTED, 0.2),
        ("Remote-state gate (post-Q R+S)", "margin, nat/answer", 0.249476, 0.055474, 0.443478, 0.5, "+0.5 excluded", AMBER, 0.8),
        ("Before-question R+S", "margin, nat/answer", 0.006221, -0.003585, 0.016027, 0.1, "+0.1 excluded", AMBER, 0.13),
    ]
    for i, (name, unit, est, a, c, t, verdict, col, span) in enumerate(rows):
        y = top2 + 45 + i * 58
        lo2, hi2 = -span * 0.6, span
        X2 = lambda v: x0 + pw * (v - lo2) / (hi2 - lo2)
        b.append(text(40, y + 2, name, 13.5, INK, weight="700"))
        b.append(text(40, y + 18, unit, 11.5, MUTED))
        b.append(line(x0, y, x0 + pw, y, GRID, 1))
        b.append(line(X2(0), y - 14, X2(0), y + 14, MUTED, 1.2))
        b.append(line(X2(t), y - 14, X2(t), y + 14, PURPLE, 1.5, "4 3"))
        b.append(text(X2(t), y - 18, f"T={t:g}", 11, PURPLE, "middle"))
        b.append(line(X2(a), y, X2(c), y, col, 4))
        b.append(circle(X2(est), y, 6, "#fff", col))
        b.append(text(x0 + pw + 12, y + 4, verdict, 13, col, weight="700"))
    b.append(text(w / 2, 575, "Exclusion ≠ “the effect is zero”. It says an effect as large as T does not fit this assay's data.", 13, RED, "middle", "600"))
    return svg(w, h, "\n".join(b))


def registry_dots():
    """Per-registry post-question vs pre-question effects (remote_components)."""
    post = [0.108780, 0.299903, 0.216186, 0.119139, 0.095315, -0.008040, 0.470604, 0.693921]
    pre = [-0.004697, -0.011947, 0.001827, 0.015817, 0.008237, 0.005807, 0.026020, 0.008703]
    regs = [f"{r:02d}" for r in range(8, 16)]
    w, h = 880, 400
    x0, y0, pw, ph = 90, 60, 620, 260
    lo, hi = -0.1, 0.75
    Y = lambda v: y0 + ph * (hi - v) / (hi - lo)
    b = [text(w / 2, 30, "Measured: eight registries, own-minus-counterfactual answer margin (nat/answer)", 16, anchor="middle", weight="700")]
    for v in (-0.1, 0, 0.1, 0.25, 0.5, 0.75):
        b.append(line(x0, Y(v), x0 + pw, Y(v), GRID if v else MUTED, 1))
        b.append(text(x0 - 8, Y(v) + 4, f"{v:+.2f}" if v else "0", 12, MUTED, "end"))
    for i, r in enumerate(regs):
        x = x0 + 40 + i * 75
        b.append(text(x, y0 + ph + 20, r, 12, MUTED, "middle"))
        b.append(circle(x - 10, Y(post[i]), 6, ORANGE))
        b.append(circle(x + 10, Y(pre[i]), 6, TEAL))
    b.append(text(x0 + pw / 2, y0 + ph + 40, "registry family (the independent unit)", 13, MUTED, "middle"))
    mp, mq = sum(post) / 8, sum(pre) / 8
    b.append(line(x0, Y(mp), x0 + pw, Y(mp), ORANGE, 1.5, "6 4"))
    b.append(line(x0, Y(mq), x0 + pw, Y(mq), TEAL, 1.5, "6 4"))
    b.append(circle(x0 + pw + 20, 90, 6, ORANGE))
    b.append(text(x0 + pw + 32, 94, "after question", 12.5))
    b.append(text(x0 + pw + 32, 110, f"mean +{mp:.3f}", 12, ORANGE))
    b.append(circle(x0 + pw + 20, 140, 6, TEAL))
    b.append(text(x0 + pw + 32, 144, "before question", 12.5))
    b.append(text(x0 + pw + 32, 160, f"mean +{mq:.3f}", 12, TEAL))
    b.append(text(w / 2, 385, "Same 8 registries, same model. Swapping R+S after the question matters a little; before it, almost not at all.", 12.5, INK, "middle", "600"))
    return svg(w, h, "\n".join(b))


def interaction():
    """Difference of differences using the held-out excess-NLL means (heldout_nll)."""
    vals = {("alpha", "1K"): 0.003026, ("alpha", "12K"): 0.003136, ("MLP", "1K"): 0.007664, ("MLP", "12K"): 0.008279}
    w, h = 860, 420
    x0, y0, ph = 90, 70, 240
    top = 0.010
    Y = lambda v: y0 + ph * (1 - v / top)
    b = [text(w / 2, 30, "Difference of differences (interaction), measured held-out means, nat/token", 17, anchor="middle", weight="700")]
    for v in (0, 0.0025, 0.005, 0.0075, 0.010):
        b.append(line(x0, Y(v), x0 + 400, Y(v), GRID))
        b.append(text(x0 - 8, Y(v) + 4, f"{v:.4f}", 11.5, MUTED, "end"))
    b.append(text(x0 - 8, y0 - 16, "excess NLL vs original", 12, MUTED))
    groups = [("alpha", BLUE, "#e6edf8", "alpha gate arm"), ("MLP", ORANGE, "#fde6d8", "MLP control arm")]
    for gi, (arm, col, fill, label) in enumerate(groups):
        gx = x0 + 30 + gi * 200
        for bi, hist in enumerate(("1K", "12K")):
            v = vals[(arm, hist)]
            x = gx + bi * 70
            b.append(rect(x, Y(v), 56, Y(0) - Y(v), fill if hist == "1K" else col, col))
            b.append(text(x + 28, Y(v) - 6, f"{v:.4f}", 11.5, INK, "middle"))
            b.append(text(x + 28, Y(0) + 16, hist, 12, MUTED, "middle"))
        b.append(text(gx + 63, Y(0) + 36, label, 13, col, "middle", "700"))
    da = vals[("alpha", "12K")] - vals[("alpha", "1K")]
    dm = vals[("MLP", "12K")] - vals[("MLP", "1K")]
    bx = 540
    b.append(rect(bx, 80, 300, 230, "#f7fafc", "#cbd5e0", 10))
    b.append(text(bx + 16, 108, "alpha: 12K − 1K", 13, BLUE, weight="700"))
    b.append(text(bx + 16, 128, f"{vals[('alpha','12K')]:.6f} − {vals[('alpha','1K')]:.6f} = {da:+.6f}", 12.5))
    b.append(text(bx + 16, 160, "MLP: 12K − 1K", 13, ORANGE, weight="700"))
    b.append(text(bx + 16, 180, f"{vals[('MLP','12K')]:.6f} − {vals[('MLP','1K')]:.6f} = {dm:+.6f}", 12.5))
    b.append(line(bx + 16, 196, bx + 284, 196, MUTED))
    b.append(text(bx + 16, 220, f"I = {da:+.6f} − ({dm:+.6f})", 13))
    b.append(text(bx + 16, 244, f"I ≈ {da - dm:+.4f} nat/token", 15, INK, weight="700"))
    b.append(text(bx + 16, 270, "positive I would mean alpha gains MORE", 12, MUTED))
    b.append(text(bx + 16, 286, "extra loss with long history than MLP", 12, MUTED))
    b.append(text(w / 2, 372, "The original-model term cancels. But note the 1K bars are NOT matched (0.0030 vs 0.0077): the dose-matching gate failed.", 12.5, RED, "middle", "600"))
    b.append(text(w / 2, 394, "Report value: −0.000506 (document-and-seed mean); computing from these rounded arm means gives −0.000505.", 12, MUTED, "middle"))
    return svg(w, h, "\n".join(b))


FIGS = {
    "decay-horizon.svg": decay_horizon,
    "softmax-nll.svg": softmax_nll,
    "nll-curve.svg": nll_curve,
    "gates.svg": gates,
    "registry-dots.svg": registry_dots,
    "interaction.svg": interaction,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in FIGS.items():
        (OUT / name).write_text(fn())
        print("wrote", OUT / name)
