"""Tool call parsing utilities.

The model emits tool calls in an XML-ish format:

<tool_call>
<function=NAME>
<parameter=ARG>VALUE</parameter>
...
</function>
</tool_call>

We parse this into structured ToolCall objects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ToolCallParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


_FUNCTION_OPEN_RE = re.compile(r"<function=([^>\n]+)>\s*", flags=re.DOTALL)
_FUNCTION_CLOSE = "</function>"

_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL,
)


def _maybe_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def parse_tool_call_block(block: str) -> ParsedToolCall:
    """Parse a single <tool_call>...</tool_call> block.

    Raises:
        ToolCallParseError: If the block is malformed.
    """
    if "<tool_call>" not in block or "</tool_call>" not in block:
        raise ToolCallParseError("Missing <tool_call> wrapper tags.")

    func_open = _FUNCTION_OPEN_RE.search(block)
    if func_open is None:
        raise ToolCallParseError("Missing <function=...> opening tag.")

    name = func_open.group(1).strip()
    func_body_start = func_open.end()
    func_body_end = block.find(_FUNCTION_CLOSE, func_body_start)
    if func_body_end == -1:
        raise ToolCallParseError("Missing </function> closing tag.")

    func_body = block[func_body_start:func_body_end]
    args: dict[str, Any] = {}
    for match in _PARAM_RE.finditer(func_body):
        param = match.group(1).strip()
        raw = match.group(2)
        # Preserve internal newlines; strip only outer newlines.
        raw = raw.strip("\n")
        args[param] = _maybe_json(raw)

    return ParsedToolCall(name=name, arguments=args)

