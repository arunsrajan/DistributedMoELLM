"""OpenAI-style tool (function) calling.

The tool list goes into the prompt through the model's own chat template
(which knows its format); the model answers with tool calls in that format,
and this module turns them back into OpenAI ``tool_calls``. Formats parsed:

* Qwen3-Coder XML::

      <tool_call>
      <function=get_weather>
      <parameter=city>
      Paris
      </parameter>
      </function>
      </tool_call>

* Qwen / Hermes JSON: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
* DeepSeek V3: ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name``
  then a JSON block, ``<｜tool▁call▁end｜>`` ... ``<｜tool▁calls▁end｜>``

XML parameters arrive as text; they are converted to the types the tool's
JSON schema declares (integer, number, boolean, object, array).
"""

from __future__ import annotations

import json
import re
import uuid

TOOL_STARTS = ("<tool_call>", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>")

_XML_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_DS_CALL = re.compile(r"<｜tool▁call▁begin｜>(?:function)?<｜tool▁sep｜>([^\n]+)\n(.*?)<｜tool▁call▁end｜>", re.S)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI request messages -> what chat templates expect: assistant tool
    call arguments as dicts (OpenAI sends JSON strings), content never None."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        elif isinstance(m["content"], list):  # content parts: keep the text
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"input": args}
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def _schemas(tools) -> dict[str, dict]:
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        if "name" in fn:
            out[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return out


def _typed(value: str, schema: dict):
    """Convert an XML parameter's text to its declared JSON-schema type."""
    kind = schema.get("type")
    if isinstance(kind, list):  # e.g. ["integer", "null"]
        kind = next((k for k in kind if k != "null"), None)
    v = value.strip()
    try:
        if kind == "integer":
            return int(v)
        if kind == "number":
            f = float(v)
            return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        if kind == "boolean":
            if v.lower() in ("true", "false"):
                return v.lower() == "true"
        if kind in ("object", "array"):
            return json.loads(v)
    except ValueError:
        pass
    if kind in (None, "null") and v.lower() == "null":
        return None
    return value


def _call(name: str, args: dict) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": name.strip(), "arguments": json.dumps(args, ensure_ascii=False)}}


def parse_tool_calls(text: str, tools=None) -> tuple[str | None, list[dict]]:
    """Split model output into (content before the calls, OpenAI tool_calls)."""
    starts = [i for i in (text.find(s) for s in TOOL_STARTS) if i >= 0]
    if not starts:
        return text, []
    cut = min(starts)
    content = text[:cut].strip() or None
    rest = text[cut:]
    schemas = _schemas(tools)
    calls = []
    for name, body in _DS_CALL.findall(rest):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
        try:
            args = json.loads(m.group(1) if m else body)
        except ValueError:
            args = {"input": body.strip()}
        calls.append(_call(name, args if isinstance(args, dict) else {"input": args}))
    for block in _TOOL_BLOCK.findall(rest):
        block = block.strip()
        funcs = _XML_FUNC.findall(block)
        if funcs:
            for name, body in funcs:
                props = schemas.get(name.strip(), {})
                args = {p.strip(): _typed(v, props.get(p.strip(), {})) for p, v in _XML_PARAM.findall(body)}
                calls.append(_call(name, args))
            continue
        try:
            obj = json.loads(block)
        except ValueError:
            continue
        for item in obj if isinstance(obj, list) else [obj]:
            if isinstance(item, dict) and "name" in item:
                args = item.get("arguments", item.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"input": args}
                calls.append(_call(item["name"], args))
    if not calls:  # looked like a call but did not parse: hand back the raw text
        return text, []
    return content, calls


def safe_prefix_len(text: str) -> int:
    """How much of ``text`` can be streamed as plain content: everything up to
    a tool-call start, or up to a trailing partial match of one."""
    starts = [i for i in (text.find(s) for s in TOOL_STARTS) if i >= 0]
    if starts:
        return min(starts)
    for k in range(min(len(text), max(map(len, TOOL_STARTS)) - 1), 0, -1):
        tail = text[-k:]
        if any(s.startswith(tail) for s in TOOL_STARTS):
            return len(text) - k
    return len(text)
