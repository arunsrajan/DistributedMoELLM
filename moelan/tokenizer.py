"""Tokenizer wrapper: HF ``tokenizers`` when installed, byte-level otherwise.

Chat templates (``chat_template.jinja`` or ``tokenizer_config.json``) are
rendered with ``jinja2`` in its sandbox, including the request's ``tools``;
without jinja2 a plain ChatML-style prompt is used.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .tools import normalize_messages


class ByteTokenizer:
    """256 byte tokens + BOS (256) + EOS (257); matches ``moelan make-tiny``."""
    bos_id, eos_id = 256, 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, keep_special: bool = False) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, keep_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=not keep_special)


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
        self.config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        jinja_file = root / "chat_template.jinja"   # newer HF layout keeps it in its own file
        template = jinja_file.read_text(encoding="utf-8") if jinja_file.exists() else self.config.get("chat_template")
        if isinstance(template, list):  # some repos store named templates
            named = {t.get("name"): t["template"] for t in template}
            template = named.get("tool_use") or named.get("default")
        self.template = template

    def encode(self, text: str) -> list[int]:
        return self.impl.encode(text)

    def decode(self, ids, keep_special: bool = False) -> str:
        return self.impl.decode(ids, keep_special)

    def chat_prompt(self, messages: list[dict], tools: list | None = None) -> str:
        messages = normalize_messages(messages)
        if self.template:
            try:
                from jinja2.exceptions import TemplateError
                from jinja2.sandbox import ImmutableSandboxedEnvironment
            except ImportError:
                pass
            else:
                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

                def raise_exception(msg):
                    raise TemplateError(msg)
                env.globals["raise_exception"] = raise_exception
                env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
                kw = {"tools": tools} if tools else {}
                return env.from_string(self.template).render(
                    messages=messages, add_generation_prompt=True,
                    bos_token=_tok_str(self.config.get("bos_token")),
                    eos_token=_tok_str(self.config.get("eos_token")), **kw)
        parts = []
        if tools:  # no template: describe the tools Hermes-style in a system turn
            spec = "\n".join(json.dumps(t.get("function", t)) for t in tools)
            parts.append("<|im_start|>system\nYou can call these functions. To call one, reply with "
                         "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>\n"
                         f"<tools>\n{spec}\n</tools><|im_end|>\n")
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        return "".join(parts) + "<|im_start|>assistant\n"


def _tok_str(t) -> str:
    if isinstance(t, dict):
        return t.get("content", "")
    return t or ""
