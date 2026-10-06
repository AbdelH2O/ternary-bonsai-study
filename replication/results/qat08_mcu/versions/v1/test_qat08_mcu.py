"""CPU tests for QAT08-MCU. Run: CUDA_VISIBLE_DEVICES= python test_qat08_mcu.py [test_name ...]

Internal restart modes (used by the parity test): --toy {full,pause,resume} [--unfreeze] --directory D
"""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_fp_map_and_payload_identity():
    import torch
    from transformers import AutoModelForCausalLM
    import qat_08b_mcu as mcu
    mapping = mcu.fp_gguf_map()
    assert len(mapping) == 169
    model = AutoModelForCausalLM.from_pretrained(ROOT / "work/qwen35_08b/base", dtype=torch.float32)
    params = mcu.fp_params(model)
    assert set(params) == {hf for hf, _, _ in mapping.values()}
    exact, template = mcu.exact_fp_values(), mcu.template_payloads()
    for gname, (hf, transform, dtype) in mapping.items():
        assert mcu.fp_payload(exact[hf], transform, dtype) == template[gname], gname


def test_hf_load_rounds_only_gdn_norms():
    """Pins a measured property of the frozen pipeline: HF loading rounds exactly the 18 GDN norms to BF16."""
    import torch
    from transformers import AutoModelForCausalLM
    import qat_08b_mcu as mcu
    model = AutoModelForCausalLM.from_pretrained(ROOT / "work/qwen35_08b/base", dtype=torch.float32)
    params, exact, template = mcu.fp_params(model), mcu.exact_fp_values(), mcu.template_payloads()
    differs = sorted(g for g, (hf, t, d) in mcu.fp_gguf_map().items() if mcu.fp_payload(params[hf], t, d) != template[g])
    assert len(differs) == 18 and all(g.endswith(".ssm_norm.weight") for g in differs), differs
    for g in differs:
        hf = mcu.fp_gguf_map()[g][0]
        assert torch.equal(params[hf].detach().float(), exact[hf].to(torch.bfloat16).float())


def _toy_setup(directory: Path, unfreeze: bool) -> None:
    import numpy as np
    (directory / "work/data").mkdir(parents=True)
    (directory / "results").mkdir()
    np.arange(512, dtype=np.uint32).tofile(directory / "work/data/train.u32")
    np.arange(512, dtype=np.uint32).tofile(directory / "work/data/monitor.u32")
    cfg = {"steps": 5, "micro_batch": 2, "grad_accum": 2, "peak_lr": .01, "warmup_steps": 2, "betas": [.9, .95],
           "grad_clip": 1., "checkpoints": [[1., "final"]], "monitor_every": 5, "resume_every_s": 1200, "fp_peak_lr": .02}
    arm = {"rule": "top86", "stream": str(directory / "work/data/train.u32"), "unfreeze": unfreeze,
           "init": "folded", "schedule": "cosine", "steps": 5}
    (directory / "results/protocol.json").write_text(json.dumps({"training": cfg, "arms": {"toy": arm}}))


def _toy_install(qat, mcu, directory: Path) -> None:
    import random
    import numpy as np
    import torch
    qat.QAT, qat.OUT, qat.DEV, qat.SEQ = directory / "work", directory / "results", "cpu", 8
    mcu.WORK, mcu.OUT, mcu.FP_COUNT, mcu.MONITOR = directory / "work", directory / "results", 1, directory / "work/data/monitor.u32"
    random.seed(321); np.random.seed(321); torch.manual_seed(321)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.arange(8).reshape(2, 4) * .03)
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.Module()
            self.model.norm.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

        def gradient_checkpointing_enable(self, **kwargs):
            pass

    def build(rule, latents=None):
        m = Toy()
        if latents:
            m.proj.weight.data.copy_(latents["projection"])
        return m, ["projection"], None

    def loss(student, teacher, ids):
        # Consumes three RNGs on purpose so restarts must restore RNG state, not only weights.
        x = ids.float().reshape(-1, 4) / 100
        noise = torch.rand_like(x) * .01 + random.random() * .001 + np.random.random() * .001
        return ((student.proj(x + noise) * (1 + student.model.norm.weight)) - .2).square().mean()

    qat.load_teacher = lambda: None
    qat.build_student, qat.distill_loss, qat.hf_name = build, loss, lambda name: "proj"
    qat.monitor_kl = lambda *args, **kwargs: 0.0


def _toy_main(mode: str, directory: Path, unfreeze: bool) -> None:
    import qat_08b as qat
    import qat_08b_mcu as mcu
    _toy_install(qat, mcu, directory)
    original = qat.AdamWBF16.step

    def step(opt, lr):
        original(opt, lr)
        if mode == "pause" and opt.params and opt.t == 2:
            (directory / "work/PAUSE_REQUESTED").write_text("{}")
    qat.AdamWBF16.step = step
    if mode == "resume":
        (directory / "work/PAUSE_REQUESTED").unlink()
    try:
        mcu.train("toy")
    except mcu.Paused:
        assert mode == "pause"
        return
    assert mode != "pause", "expected an optimizer-boundary pause"


def _kl_rows(directory: Path) -> list[float]:
    rows = [json.loads(l) for l in (directory / "work/toy/train.jsonl").read_text().splitlines()]
    return [r["train_kl"] for r in rows if "train_kl" in r]


def test_trainer_matches_frozen_loop():
    import torch
    import qat_08b as qat
    import qat_08b_mcu as mcu
    with tempfile.TemporaryDirectory() as name:
        a, b = Path(name) / "frozen", Path(name) / "mcu"
        a.mkdir(); b.mkdir(); _toy_setup(a, False); _toy_setup(b, False)
        _toy_install(qat, mcu, a)
        qat.train("toy")
        _toy_install(qat, mcu, b)
        mcu.train("toy")
        x = torch.load(a / "work/toy/resume.pt", weights_only=False)
        y = torch.load(b / "work/toy/resume.pt", weights_only=False)
        assert x["step"] == y["step"] == 5 and x["opt"]["t"] == y["opt"]["t"] == 5
        assert torch.equal(x["latents"]["projection"], y["latents"]["projection"])
        assert all(torch.equal(p, q) for k in ("m", "v") for p, q in zip(x["opt"][k], y["opt"][k]))
        assert _kl_rows(a) == _kl_rows(b)
        assert torch.equal(torch.load(a / "work/toy/latent_final.pt")["projection"],
                           torch.load(b / "work/toy/latent_final.pt")["projection"])


def test_unfreeze_pause_resume_parity():
    import torch
    with tempfile.TemporaryDirectory() as name:
        full, split = Path(name) / "full", Path(name) / "split"
        full.mkdir(); split.mkdir(); _toy_setup(full, True); _toy_setup(split, True)
        run = lambda mode, d: subprocess.run([sys.executable, str(Path(__file__)), "--toy", mode, "--unfreeze",
                                              "--directory", str(d)], check=True)
        run("full", full)
        run("pause", split)
        paused = torch.load(split / "work/toy/resume.pt", weights_only=False)
        assert paused["step"] == 2 and not (split / "work/toy/DONE").exists()
        run("resume", split)  # fresh process
        x = torch.load(full / "work/toy/resume.pt", weights_only=False)
        y = torch.load(split / "work/toy/resume.pt", weights_only=False)
        assert x["step"] == y["step"] == 5
        assert torch.equal(x["latents"]["projection"], y["latents"]["projection"])
        assert torch.equal(x["fp"]["model.norm.weight"], y["fp"]["model.norm.weight"])
        assert not torch.equal(x["fp"]["model.norm.weight"], torch.zeros(2)), "FP tensor must actually train"
        for key in ("opt", "opt_fp"):
            assert x[key]["t"] == y[key]["t"] == 5
            assert all(torch.equal(p, q) for k in ("m", "v") for p, q in zip(x[key][k], y[key][k]))
        assert _kl_rows(full) == _kl_rows(split)
        assert torch.equal(torch.load(full / "work/toy/fp_final.pt")["model.norm.weight"], x["fp"]["model.norm.weight"])


PARA = ("The river carried sediment toward the delta every spring, and farmers planted barley along the fertile banks "
        "while merchants traded copper and textiles in the bustling harbour near the old fortress walls. Historians "
        "believe the settlement flourished because its granaries protected harvests against famine during unusually "
        "severe winters that followed volcanic eruptions far away. Archaeologists recently uncovered pottery, jewellery "
        "and inscriptions describing elaborate festivals that celebrated the returning floods and honoured the protective "
        "goddess of the waters with music, dancing and processions throughout several consecutive nights every year.")


def test_blank_item_properties():
    import random
    import re
    import prepare_qat08_mcu as p
    assert 80 <= len(PARA.split()) <= 200
    for seed in range(20):
        item = p.blank_item(PARA, random.Random(seed))
        assert item is not None
        sentence, target, distractors = item["sentence"], item["target"], item["distractors"]
        assert sentence.count("____") == 1 and sentence.replace("____", target) in PARA
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", PARA)]
        assert words.count(target) == 1
        assert len(set(distractors)) == 3 and target not in distractors
        for d in distractors:
            assert words.count(d) == 1 and not re.search(rf"\b{d}\b", sentence)
    assert p.blank_item("too short to use", random.Random(0)) is None


def test_incontext_balance():
    import collections
    import re
    import prepare_qat08_mcu as p
    from prepare_qat08_chat import tokenizer
    tok = tokenizer()
    items = p.incontext_items(tok, 40, seed=7)
    assert collections.Counter(i["gold"] for i in items) == {"A": 10, "B": 10, "C": 10, "D": 10}
    for it in items:
        assert p.PASSAGE_SEQS[0] <= it["seq"] < p.PASSAGE_SEQS[1]
        passage = it["prompt"].split("Passage:\n", 1)[1].split("\n\nWhich word fills the blank?", 1)[0]
        sentence = it["prompt"].split("Which word fills the blank?\n", 1)[1].split("\n\n", 1)[0]
        options = dict(re.findall(r"^([ABCD])\. (\w+)$", it["prompt"], re.M))
        assert sentence.replace("____", options[it["gold"]]) in passage
        assert sum(sentence.replace("____", w) in passage for w in options.values()) == 1
        assert len(it["prompt_ids"]) <= 768 and it["prompt_ids"][-9:] == [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]


def test_screen_near_duplicate():
    import prepare_qat08_mcu as p
    evals = ["Kimberly bought 8 packages of cat food and 6 packages of dog food. Each package of cat food contained 11 tins, "
             "and each package of dog food contained 6 tins. How many more tins of cat food than dog food did Kimberly buy?",
             "Which of the following is a prime number?\n\nA. 4\nB. 6\nC. 7\nD. 9",
             "The three angles in a triangle add up to 180 degrees."]
    twin = {"id": "t", "source": "arc", "question": "Chad bought 6 packages of cat food and 2 packages of dog food. Each package "
            "of cat food contained 9 cans, and each package of dog food contained 3 cans. How many more cans of cat food than "
            "dog food did Chad buy?", "prompt": "x"}
    exact = {"id": "e", "source": "arc", "question": "Which of the following is a prime number?", "prompt": "x"}
    other = {"id": "o", "source": "arc", "question": "Which gas do plants absorb from the atmosphere during photosynthesis?", "prompt": "x"}
    kept, audit = p.screen([twin, exact, other], evals)
    assert [k["id"] for k in kept] == ["o"]
    assert audit["dropped"] == {"near_duplicate": 1, "exact_question": 1}
    assert kept[0]["max_eval_cosine"] < 0.6


def test_stream_m_layout():
    import numpy as np
    import prepare_qat08_mcu as p
    with tempfile.TemporaryDirectory() as name:
        d = Path(name)
        p.SEQ = 4
        try:
            chat = np.arange(32 * 4, dtype=np.uint32).reshape(32, 4) + 10_000
            chat.tofile(d / "c.u32")
            fine = np.arange(200 * 4, dtype=np.uint32) + 50_000
            fine.tofile(d / "f.u32")
            mc = np.arange(10, dtype=np.uint32) + 90_000
            info = p.stream_m(mc, d / "m.u32", d / "c.u32", d / "f.u32", n=32)
            out = np.fromfile(d / "m.u32", dtype=np.uint32).reshape(32, 4)
            assert info == {"fineweb_seqs": 22, "mc_seqs": 2, "chat_seqs": 8}
            for s in range(32):
                if s % 4 == 3:
                    assert np.array_equal(out[s], chat[s])
            assert np.array_equal(out[14], mc[np.arange(4) % 10]) and np.array_equal(out[30], mc[(4 + np.arange(4)) % 10])
            fine_rows = [s for s in range(32) if s % 4 != 3 and s % 16 != 14]
            assert all(np.array_equal(out[s], fine[i * 4:(i + 1) * 4]) for i, s in enumerate(fine_rows))
        finally:
            p.SEQ = 1024


def test_stream_b_continues_c():
    import numpy as np
    import prepare_qat08_mcu as p
    with tempfile.TemporaryDirectory() as name:
        d = Path(name)
        p.SEQ = 4
        try:
            fine = np.arange(200 * 4, dtype=np.uint32)
            fine.tofile(d / "f.u32")
            unique = np.arange(13, dtype=np.uint32) + 7_000
            info = p.stream_b(unique, d / "b.u32", d / "f.u32", n=8, fine_start=5, chat_start=3)
            out = np.fromfile(d / "b.u32", dtype=np.uint32).reshape(8, 4)
            assert np.array_equal(out[0], fine[20:24]) and np.array_equal(out[4], fine[32:36])
            assert np.array_equal(out[3], unique[(12 + np.arange(4)) % 13]) and np.array_equal(out[7], unique[(16 + np.arange(4)) % 13])
            assert info == {"fineweb_range": [5, 11], "chat_window_range": [3, 5]}
        finally:
            p.SEQ = 1024


def test_c_stream_offsets_match_b_start():
    """The real chat stream ends at FineWeb sequence 47,999 and chat window 15,999, so B starts at 48,000 / 16,000."""
    import numpy as np
    import prepare_qat08_mcu as p
    c = np.memmap(p.CHAT_DATA / "train.u32", dtype=np.uint32, mode="r").reshape(-1, p.SEQ)
    unique = np.fromfile(p.CHAT_DATA / "chat_unique.u32", dtype=np.uint32)
    fine = np.memmap(p.FINEWEB, dtype=np.uint32, mode="r")
    assert np.array_equal(c[63999], unique[(15999 * p.SEQ + np.arange(p.SEQ)) % len(unique)])
    assert np.array_equal(c[63998], fine[47999 * p.SEQ:48000 * p.SEQ])


def test_book_inventory_finds_every_historical_form():
    import prepare_qat08_mcu as p
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "a.py").write_text('GUTENBERG = {"dict_int": 11, "dict_tuple": (12, "Dict Tuple")}\n')
        (root / "b.py").write_text('CALIBRATION_BOOKS = ("calbook.txt",)\nNEW_HELDOUT_BOOKS = ("held_one.txt",)\n')
        (root / "c.py").write_text('SPLIT = {"split_book.txt": "validation"}\n')
        (root / "d.py").write_text('DOCS = [("doc_book", 13, "Doc Book Title")]\nHELDOUT = ("bare_stem",)\n')
        (root / "e").mkdir()
        (root / "e/protocol.json").write_text(json.dumps({"documents": [
            {"id": "json_book", "gutenberg_ebook": 14, "source_url": "https://www.gutenberg.org/cache/epub/15/pg15.txt"}]}))
        (root / "e/manifest.json").write_text(json.dumps({"books": {"manifest_book": {"gutenberg_id": 16}}}))
        (root / "g/books").mkdir(parents=True)
        (root / "g/books/disk_book-25.txt").write_text("x")
        (root / "g/books/ids_book.ids").write_text("1")
        (root / "report.md").write_text("We scored The Title Only Novel last week.\n")
        (root / "mine.py").write_text('GUTENBERG = {"clean": 99}\n')
        inv = p.book_inventory(root, exclude={root / "mine.py"})
        books = {"dict_int": (11, "Other A"), "dict_tuple": (900, "Dict Tuple"), "calbook": (901, "X1"),
                 "held_one": (902, "X2"), "split_book": (903, "X3"), "doc_book": (904, "Y4"), "bare_stem": (905, "Y5"),
                 "fresh_a": (14, "Y6"), "fresh_b": (15, "Y7"), "fresh_c": (16, "Y8"), "json_book": (906, "Y9"),
                 "manifest_book": (907, "Y10"), "disk_book": (908, "Y11"), "ids_book": (909, "Y12"),
                 "novel": (910, "The Title Only Novel"), "variant": (911, "Doc Book Title; or, a Subtitle"),
                 "clean": (99, "Clean Unused Book")}
        clashes = p.book_clashes(books, inv)
        assert set(clashes) == set(books) - {"clean"}, sorted(set(books) - set(clashes))


def test_pre_freeze_incidents_are_carried():
    import prepare_qat08_mcu as p
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "pre_freeze_incidents.json"
        assert p.pre_freeze_incidents(path) == []
        p.record_incident({"incident": "first"}, path)
        p.record_incident({"incident": "second"}, path)
        assert [r["incident"] for r in p.pre_freeze_incidents(path)] == ["first", "second"]
        assert all("time" in r for r in p.pre_freeze_incidents(path))
    import freeze_qat08_mcu as f
    assert "pre_freeze_incidents" in inspect.getsource(f.design)


def test_new_books_are_unused():
    import prepare_qat08_mcu as p
    inv = p.book_inventory()
    previously_reused = {"emma": (158, "Emma"), "little_women": (514, "Little Women"), "middlemarch": (145, "Middlemarch"),
                         "wuthering_heights": (768, "Wuthering Heights"), "monte_cristo": (1184, "The Count of Monte Cristo"),
                         "dracula": (345, "Dracula"), "pride_and_prejudice": (1342, "Pride and Prejudice"),
                         "frankenstein": (84, "Frankenstein"), "war_of_the_worlds": (36, "The War of the Worlds")}
    assert set(p.book_clashes(previously_reused, inv)) == set(previously_reused), "regression: missed inventory forms"
    assert not p.book_clashes(p.GUTENBERG, inv)
    assert not p.book_clashes(p.RESERVE, inv)
    assert set(p.VALIDATION) <= set(p.GUTENBERG)


def test_select_arms_budget_rule():
    import freeze_qat08_mcu as f
    assert f.select_arms({"arms_required": ["M", "U"], "arms_optional": ["MU"]}, prep_hours=1.6) == ["M", "U", "MU"]
    assert f.select_arms({"arms_required": ["M", "U", "B"], "arms_optional": ["MU"]}, prep_hours=1.6) == ["M", "U", "B"]
    assert f.select_arms({"arms_required": ["U", "B"], "arms_optional": []}, prep_hours=1.6) == ["U", "B"]


def test_choose_fp_lr():
    import freeze_qat08_mcu as f
    rec = lambda lr, kl: {"fp_lr": lr, "curve": [{"step": 0, "monitor_kl": 8.3}, {"step": 120, "monitor_kl": kl}]}
    out = f.choose_fp_lr([rec(0.0, 1.20), rec(1e-5, 1.19), rec(1e-4, 1.15), rec(1e-3, 1.31)])
    assert out["selected_lr"] == 1e-4 and out["baseline_frozen_kl"] == 1.20 and out["unfreeze_helps_in_probe"] is True
    out = f.choose_fp_lr([rec(0.0, 1.10), rec(1e-5, 1.12), rec(1e-4, 1.15), rec(1e-3, 1.31)])
    assert out["selected_lr"] == 1e-5 and out["unfreeze_helps_in_probe"] is False


def test_decide_mcu():
    import freeze_qat08_mcu as f
    import score_qat08_mcu as s
    def arm(effect=0.0, effect_lo=-1.0, books=0.0, books_hi=0.02, mmlu_lo=22.0, gsm_lo=12.0, strict_lo=12.0,
            fresh_lo=22.0, fresh_pw_lo=49.0, bind=95.0, retr=94.0, gsm_diff_lo=-1.0, mass=0.8, pw=50.0):
        return {"mmlu_pairwise_vs_C": {"mean": effect, "lo": effect_lo}, "books_vs_C": {"mean": books, "hi": books_hi},
                "mmlu": {"lo": mmlu_lo}, "gsm8k": {"lo": gsm_lo}, "gsm8k_strict": {"lo": strict_lo},
                "mmlu_fresh": {"lo": fresh_lo}, "mmlu_fresh_pairwise_lo": fresh_pw_lo, "binding_calibrated": bind,
                "retrieval_accuracy": retr, "gsm8k_vs_C": {"lo": gsm_diff_lo}, "letter_mass": mass, "mmlu_pairwise": pw}
    none = {"arms": {"M": arm(), "U": arm()}}
    assert s.decide_mcu(none, f.DECISION, "readout_broken", ["M", "U"])["recommendation"] == "stop_data_iteration_0p8b"
    assert s.decide_mcu(none, f.DECISION, "readout_broken", ["M", "U"])["arms"]["M"]["format_only"] is True
    m_effect = {"arms": {"M": arm(effect=6.0, effect_lo=2.0, fresh_pw_lo=53.0, pw=56.0), "U": arm()}}
    d = s.decide_mcu(m_effect, f.DECISION, "readout_broken", ["M", "U"])
    assert d["arms"]["M"]["H_M"] is True and d["recommendation"] == "iterate_0p8b_mc_data"
    win = {"arms": {"M": arm(effect=20.0, effect_lo=15.0, mmlu_lo=31.0, fresh_lo=31.0, fresh_pw_lo=60.0, pw=65.0), "U": arm()}}
    d = s.decide_mcu(win, f.DECISION, "readout_broken", ["M", "U"])
    assert d["recommendation"] == "rent_1p7b_with_recipe" and d["recipe_arms"] == ["M"]
    costly = {"arms": {"M": arm(effect=20.0, effect_lo=15.0, mmlu_lo=31.0, fresh_lo=31.0, fresh_pw_lo=60.0, pw=65.0, books_hi=0.2)}}
    assert s.decide_mcu(costly, f.DECISION, "readout_broken", ["M"])["recommendation"] == "iterate_0p8b_mc_data"
    books_only = {"arms": {"U": arm(), "B": arm(books=-0.08, books_hi=-0.03)}}
    d = s.decide_mcu(books_only, f.DECISION, "knowledge_lost", ["U", "B"])
    assert d["arms"]["B"]["H_B"] is True and d["recommendation"] == "iterate_0p8b"  # book gain alone never triggers rental
    knowledge = {"arms": {"U": arm(), "B": arm(effect=5.0, effect_lo=1.0, pw=55.0)}}
    assert s.decide_mcu(knowledge, f.DECISION, "knowledge_lost", ["U", "B"])["recommendation"] == "rent_1p7b_token_ladder"
    fresh = {"arms": {"U": arm(), "B": arm(fresh_pw_lo=53.0)}}
    assert s.decide_mcu(fresh, f.DECISION, "knowledge_lost", ["U", "B"])["recommendation"] == "rent_1p7b_token_ladder"
    u_helps = {"arms": {"U": arm(effect=5.0, effect_lo=1.0, pw=55.0), "B": arm(effect=5.0, effect_lo=1.0, pw=55.0)}}
    assert s.decide_mcu(u_helps, f.DECISION, "knowledge_lost", ["U", "B"])["recommendation"] == "iterate_0p8b"
    assert s.decide_mcu(knowledge, f.DECISION, "ambiguous", ["U", "B"])["recommendation"] == "iterate_0p8b"


def test_supervisor_accounting_counts_dangling_stages():
    import qat08_mcu_control as c
    with tempfile.TemporaryDirectory() as name:
        c.WORK = Path(name)
        rows = [{"event": "sv_start", "phase": "run", "stage": "train-M", "time": 1000.0},
                {"event": "sv_heartbeat", "phase": "run", "stage": "train-M", "time": 2200.0},
                {"event": "sv_finish", "phase": "run", "stage": "train-M", "time": 4600.0, "seconds": 3600.0},
                {"event": "sv_start", "phase": "run", "stage": "train-MU", "time": 5000.0},
                {"event": "sv_heartbeat", "phase": "run", "stage": "train-MU", "time": 6800.0},
                {"event": "sv_start", "phase": "prep", "stage": "pilot", "time": 90000.0},
                {"event": "sv_finish", "phase": "prep", "stage": "pilot", "time": 90360.0, "seconds": 360.0}]
        (c.WORK / "execution.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert abs(c._used() - (3600 + 1800 + 360) / 3600) < 1e-9          # MU died after its last heartbeat
        assert abs(c._used(stage="train-M") - 1.0) < 1e-9                  # exact stage match, not a prefix
        assert abs(c._used(stage="train-MU") - 0.5) < 1e-9
        assert abs(c._used(phase="prep") - 0.1) < 1e-9


def test_supervisor_wait_hard_kills_at_budget():
    import time
    import qat08_mcu_control as c
    stubborn = subprocess.Popen([sys.executable, "-c", "import signal,time\nfor s in (signal.SIGUSR1, signal.SIGTERM):\n    signal.signal(s, signal.SIG_IGN)\ntime.sleep(60)"])
    t0 = time.time()
    outcome, rc = c._wait(stubborn, deadline=time.time() + 1.0, grace=1.0, beat=lambda: None, interval=0.2)
    assert outcome == "budget" and stubborn.poll() is not None and time.time() - t0 < 10
    quick = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    assert c._wait(quick, deadline=time.time() + 30, grace=1.0, beat=lambda: None, interval=0.2) == ("exited", 3)


def test_scoring_pause_wrapper():
    import qat08_mcu_control as c
    with tempfile.TemporaryDirectory() as name:
        c.WORK, c.PAUSE = Path(name), Path(name) / "PAUSE_REQUESTED"
        rt = c._runtime()
        original = subprocess.run
        try:
            c.install_scoring_pause(rt)
            assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
            c.PAUSE.write_text("{}")
            try:
                subprocess.run([sys.executable, "-c", "pass"])
            except rt.Paused:
                pass
            else:
                raise AssertionError("a scorer launch must honour a pause request")
        finally:
            subprocess.run = original
            rt._requested = False


def test_contamination_rerun_after_interruption():
    import numpy as np
    import freeze_qat08_mcu as f
    with tempfile.TemporaryDirectory() as name:
        d = Path(name)
        stream = d / "s.u32"
        np.arange(400, dtype=np.uint32).tofile(stream)
        f.OUT, original = d, f.read_cases
        f.read_cases = lambda: {"x": [("a", np.arange(100, 130, dtype=np.uint32)), ("b", np.arange(5000, 5030, dtype=np.uint32))]}
        try:
            (d / "contamination_s.jsonl").write_text('{"set": "x", "id": "a", "trunc')  # interrupted earlier attempt
            out = f.contamination(stream)
            assert out["sets"]["x"]["cases_with_majority_13gram_overlap"] == 1
            assert len((d / "contamination_s.jsonl").read_text().splitlines()) == 2
        finally:
            f.read_cases = original


def test_pack_rerun_after_interruption():
    import numpy as np
    import prepare_qat08_mcu as p
    from prepare_qat08_chat import tokenizer
    tok = tokenizer()
    with tempfile.TemporaryDirectory() as name:
        d = Path(name)
        saved = (p.DATA, p.OUT, p.MIN_KEPT, p.MIN_UNIQUE, p.MAX_PASSES, p.MC_TOKENS, p.eval_contents)
        p.DATA, p.OUT = d / "data", d
        p.DATA.mkdir()
        p.MIN_KEPT, p.MIN_UNIQUE, p.MAX_PASSES, p.MC_TOKENS = 1, 10, 1000.0, 2048
        p.eval_contents = lambda tok: []
        try:
            (d / "design.json").write_text(json.dumps({"arms": {}}))
            prompt = "The following is a multiple choice question about science. Reply with just the letter of the correct answer.\n\nWhat is H2O?\n\nA. water\nB. salt\nC. iron\nD. gold"
            import freeze_17b as fz
            rows = [{"id": "arc-x-0", "source": "arc", "gold": "A", "reasoning": False, "prompt": prompt, "prompt_ids": fz.chat(tok, prompt)}]
            (p.DATA / "mc_prompts.jsonl").write_text(json.dumps(rows[0]) + "\n")
            (p.DATA / "mc_teacher.jsonl").write_text(json.dumps({"id": "arc-x-0", "text": "A", "tokens": 1, "stop_type": "eos"}) + "\n")
            (p.DATA / "kept_mc.json").write_text("[{\"id\": \"arc-x-0\"")   # interrupted earlier pack
            p.pack()
            record = json.loads((p.DATA / "data_record.json").read_text())
            assert record["mc_kept"] == 1 and record["teacher_letter_accuracy"] == {"arc_correct": 1}
            assert json.loads((p.DATA / "kept_mc.json").read_text())[0]["id"] == "arc-x-0"
        finally:
            p.DATA, p.OUT, p.MIN_KEPT, p.MIN_UNIQUE, p.MAX_PASSES, p.MC_TOKENS, p.eval_contents = saved


def test_fresh_eligible_excludes_every_redux_item():
    import prepare_qat08_mcu as p
    from prepare_qat08_chat import normalize
    item = lambda q, opts=("a1", "b1", "c1", "d1"): {"question": q, "choices": list(opts), "answer": 0, "subject": "x"}
    table = [item("Keep me please"), item("A flagged Redux question"), item("An ok Redux question"), item("Dev question")]
    eligible, dropped = p.fresh_eligible(table, banned=set(), redux_questions={normalize("An ok Redux question")},
                                         redux_grams=set(), redux_options=set(), dev={normalize("Dev question")},
                                         redux_all={normalize("A flagged Redux question"), normalize("An ok Redux question")})
    assert [i for i, _ in eligible] == [0]
    assert sum(dropped.values()) == 3


def test_runtime_and_environment_checks():
    import freeze_qat08_mcu as f
    f.check_runtime_matches_chat()
    env = f.current_environment()
    f.check_environment(env)
    bad = {**env, "packages": {**env["packages"], "transformers": "0.0.0"}}
    try:
        f.check_environment(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("a changed transformers version must be rejected")


def test_design_deviations_are_complete():
    import freeze_qat08_mcu as f
    text = " ".join(f.DEVIATIONS).lower()
    for needle in ("openbookqa", "protocol_approval", "constant lr", "gdn norm", "joint", "25", "single option order",
                   "512", "in_proj", "48", "decision_amendment"):
        assert needle in text, needle


def test_controller_stages():
    import qat08_mcu_control as c
    assert c.stages("prep", decision={"arms_required": ["M", "U"], "arms_optional": ["MU"]}) == \
        ["selftest", "pilot", "probe-0", "probe-1e-05", "probe-0.0001", "probe-0.001"]
    assert c.stages("prep", decision={"arms_required": ["M"], "arms_optional": []}) == ["selftest", "pilot"]
    assert c.stages("data") == ["generate", "pack", "protocol"]
    p = {"arms": {"M": {"init": "folded"}, "B": {"init": "work/qat08_chat/qat_chat_42/resume.pt"}},
         "training": {"checkpoints": [[0.125, "t8m"], [0.25, "t16m"], [0.5, "t33m"], [1.0, "final"]]}}
    run = c.stages("run", p=p)
    assert run[:2] == ["train-M", "train-B"]
    assert "export-M-init" in run and "export-B-init" not in run and "export-B-final" in run
    assert run[-5:] == ["curve", "heldout", "gsm8k", "analyze", "report"]


def test_worker_runs_selftest_stage():
    import inspect
    import qat08_mcu_control as c
    assert 'name == "selftest"' in inspect.getsource(c.worker) and "mcu.selftest()" in inspect.getsource(c.worker)


def test_controller_approval_binding():
    import qat08_mcu_control as c
    with tempfile.TemporaryDirectory() as name:
        c.OUT = Path(name)
        (c.OUT / "design.json").write_text("{}")
        (c.OUT / "protocol.json").write_text('{"x": 1}')
        from score_17b import sha
        (c.OUT / "protocol_approval.json").write_text(json.dumps({"approved_by_user": True, "design_sha256": sha(c.OUT / "design.json"),
                                                                  "protocol_sha256": "0" * 64}))
        try:
            c.check_approval("run")
        except AssertionError:
            pass
        else:
            raise AssertionError("approval bound to a different protocol must be rejected")
        (c.OUT / "protocol_approval.json").write_text(json.dumps({"approved_by_user": True, "design_sha256": sha(c.OUT / "design.json"),
                                                                  "protocol_sha256": sha(c.OUT / "protocol.json")}))
        c.check_approval("run")


def _diag_dir(diag: Path, amended: bool = True) -> dict:
    """A minimal consistent diagnostics directory: plan, results, decision bound to both, optional amendment."""
    from score_17b import sha
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "plan.json").write_text('{"plan": 1}')
    (diag / "results.json").write_text('{"results": 1}')
    (diag / "decision.json").write_text(json.dumps({"branch": "harness_invalid", "arms_required": [], "arms_optional": [],
                                                    "stop": True, "results_sha256": sha(diag / "results.json"),
                                                    "plan_sha256": sha(diag / "plan.json")}))
    record = {"amends_decision_sha256": sha(diag / "decision.json"), "results_sha256": sha(diag / "results.json"),
              "plan_sha256": sha(diag / "plan.json"),
              "amended": {"branch": "readout_broken", "arms_required": ["M", "U", "B"], "arms_optional": ["MU"], "stop": False}}
    if amended:
        (diag / "decision_amendment.json").write_text(json.dumps(record))
    return record


def _rejects(diag: Path, why: str) -> None:
    import qat08_mcu_control as c
    try:
        c.diagnostics(diag)
    except AssertionError:
        return
    raise AssertionError(why)


def test_diagnostics_reads_hash_bound_amendment():
    import qat08_mcu_control as c
    from score_17b import sha
    with tempfile.TemporaryDirectory() as name:
        diag = Path(name)
        record = _diag_dir(diag, amended=False)
        decision, prov = c.diagnostics(diag)
        assert decision["stop"] is True and prov == {"decision_sha256": sha(diag / "decision.json"), "amendment_sha256": None}
        (diag / "decision_amendment.json").write_text(json.dumps(record))
        decision, prov = c.diagnostics(diag)
        assert decision["branch"] == "readout_broken" and decision["stop"] is False and decision["amended"] is True
        assert prov["amendment_sha256"] == sha(diag / "decision_amendment.json")
        assert c.stages("prep", decision=decision) == ["selftest", "pilot", "probe-0", "probe-1e-05", "probe-0.0001", "probe-0.001"]
        (diag / "decision_amendment.json").write_text(json.dumps({**record, "amends_decision_sha256": "0" * 64}))
        _rejects(diag, "an amendment bound to a different decision must be rejected")


def test_diagnostics_rejects_mutated_provenance():
    for amended in (False, True):
        for target, why in (("plan.json", "current plan mutated"), ("results.json", "current results mutated")):
            with tempfile.TemporaryDirectory() as name:
                diag = Path(name)
                _diag_dir(diag, amended)
                (diag / target).write_text((diag / target).read_text() + " ")
                _rejects(diag, f"{why} (amended={amended}) must be rejected")
        with tempfile.TemporaryDirectory() as name:
            diag = Path(name)
            _diag_dir(diag, amended)
            d = json.loads((diag / "decision.json").read_text())
            (diag / "decision.json").write_text(json.dumps({**d, "branch": "readout_broken", "stop": False}))
            if amended:
                _rejects(diag, "a decision changed after the amendment must be rejected")
            else:
                import qat08_mcu_control as c
                c.diagnostics(diag)  # a consistent unamended decision still loads
            (diag / "decision.json").write_text(json.dumps({**d, "results_sha256": "0" * 64}))
            _rejects(diag, f"a decision bound to other results (amended={amended}) must be rejected")
            (diag / "decision.json").write_text(json.dumps({**d, "plan_sha256": "0" * 64}))
            _rejects(diag, f"a decision bound to another plan (amended={amended}) must be rejected")


def test_diagnostics_real_files_and_plan_mutation():
    import shutil
    import qat08_mcu_control as c
    real = Path(__file__).resolve().parent / "results/diag_readout"
    names = ("plan.json", "results.json", "decision.json", "decision_amendment.json")
    if not all((real / n).exists() for n in names):
        return
    with tempfile.TemporaryDirectory() as name:
        diag = Path(name)
        for n in names:
            shutil.copyfile(real / n, diag / n)
        decision, _ = c.diagnostics(diag)
        assert decision["branch"] == "readout_broken"
        (diag / "plan.json").write_text((diag / "plan.json").read_text() + " ")
        _rejects(diag, "copied real files with an altered plan must be rejected")


def test_prep_approval_binds_amendment():
    import qat08_mcu_control as c
    from score_17b import sha
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        c.OUT, c.WORK, c.DIAG = root / "out", root / "work", root / "diag"
        for d in (c.OUT, c.WORK / "data", c.DIAG):
            d.mkdir(parents=True)
        (c.OUT / "cases_record.json").write_text("{}")
        (c.WORK / "data/mc_record.json").write_text("{}")
        _diag_dir(c.DIAG)
        base = {"approved_by_user": True, "cases_record_sha256": sha(c.OUT / "cases_record.json"),
                "mc_record_sha256": sha(c.WORK / "data/mc_record.json"), "decision_sha256": sha(c.DIAG / "decision.json")}
        (c.OUT / "prep_approval.json").write_text(json.dumps(base))
        try:
            c.check_approval("prep")
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError("prep approval must bind the amendment when one exists")
        (c.OUT / "prep_approval.json").write_text(json.dumps({**base, "amendment_sha256": sha(c.DIAG / "decision_amendment.json")}))
        c.check_approval("prep")


TESTS = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="*")
    ap.add_argument("--toy", choices=["full", "pause", "resume"])
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--directory")
    a = ap.parse_args()
    if a.toy:
        _toy_main(a.toy, Path(a.directory), a.unfreeze)
    else:
        for t in TESTS:
            if not a.tests or t.__name__ in a.tests:
                t()
                print("PASS", t.__name__, flush=True)
