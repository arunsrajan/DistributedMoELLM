"""Tokenizer wrapper: HF ``tokenizers`` when installed, byte-level otherwise.

Chat templates from ``tokenizer_config.json`` are rendered with ``jinja2`` when
available; otherwise a plain ChatML-style prompt is used.
"""

from __future__ import annotations

import json
from pathlib import Path


class ByteTokenizer:
    """256 byte tokens + BOS (256) + EOS (257); matches ``moelan make-tiny``."""
    bos_id, eos_id = 256, 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)


class Tokenizer:
    def __init__(self, model_dir: str | Path):
        root = Path(model_dir)
        tj = root / "tokenizer.json"
        if tj.exists():
            try:
                self.impl = HFTokenizer(tj)
            except ImportError as exc:
                raise SystemExit("this model needs the 'tokenizers' package: pip install tokenizers") from exc
        else:
            self.impl = ByteTokenizer()
        cfg_path = root / "tokenizer_config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    def encode(self, text: str) -> list[int]:
        return self.impl.encode(text)

    def decode(self, ids) -> str:
        return self.impl.decode(ids)

    def chat_prompt(self, messages: list[dict]) -> str:
        template = self.config.get("chat_template")
        if isinstance(template, list):  # some repos store named templates
            template = next((t["template"] for t in template if t.get("name") == "default"), None)
        if template:
            try:
                import jinja2
                env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

                def raise_exception(msg):
                    raise jinja2.TemplateError(msg)
                env.globals["raise_exception"] = raise_exception
                return env.from_string(template).render(
                    messages=messages, add_generation_prompt=True,
                    bos_token=_tok_str(self.config.get("bos_token")),
                    eos_token=_tok_str(self.config.get("eos_token")))
            except ImportError:
                pass
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(parts) + "<|im_start|>assistant\n"


def _tok_str(t) -> str:
    if isinstance(t, dict):
        return t.get("content", "")
    return t or ""
