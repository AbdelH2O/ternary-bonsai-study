"""Render the GGUF's actual chat template and count tokens with Prism llama-tokenize."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import jinja2

from inspect_gguf import HeaderReader


MODEL = Path("models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf")
TOKENIZER = Path("bin/cuda/llama-tokenize")
COUNT_PATTERN = re.compile(r"Total number of tokens: (\d+)")


def chat_template(model: Path = MODEL):
    with model.open("rb") as stream:
        reader = HeaderReader(stream)
        if stream.read(4) != b"GGUF":
            raise ValueError("Not a GGUF")
        reader.scalar("I")
        reader.scalar("Q")
        metadata_count = reader.scalar("Q")
        template = None
        for _ in range(metadata_count):
            key, kind = reader.string(), reader.scalar("I")
            if key == "tokenizer.chat_template":
                template = reader.value(kind)
            else:
                reader.value(kind, False)
    if template is None:
        raise ValueError("GGUF has no chat template")
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda reason: (_ for _ in ()).throw(ValueError(reason))
    return env.from_string(template)


def render_chat(template, prompt: str) -> str:
    return template.render(messages=[{"role": "user", "content": prompt}],
                           add_generation_prompt=True, enable_thinking=False, tools=None)


def count_tokens(text: str, model: Path = MODEL, tokenizer: Path = TOKENIZER) -> int:
    result = subprocess.run([str(tokenizer), "-m", str(model), "--stdin", "--show-count", "--no-bos"],
                            input=text.encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    # llama-tokenize prints raw token pieces, which need not form valid UTF-8.
    match = COUNT_PATTERN.search(result.stdout.decode("utf-8", errors="replace"))
    if not match:
        raise ValueError(f"Tokenizer produced no count: {result.stderr[-500:]!r}")
    return int(match.group(1))
