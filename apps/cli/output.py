from __future__ import annotations

import json
from typing import Any, Iterable, Sequence


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def format_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows_list = [list(r) for r in rows]
    widths = [len(h) for h in headers]
    for r in rows_list:
        for i, cell in enumerate(r):
            if i >= len(widths):
                break
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: Sequence[str]) -> str:
        padded = []
        for i, c in enumerate(cols):
            if i >= len(widths):
                padded.append(c)
            else:
                padded.append(c.ljust(widths[i]))
        return "  ".join(padded).rstrip()

    out = [fmt_row(list(headers)), fmt_row(["-" * w for w in widths])]
    out.extend(fmt_row(r) for r in rows_list)
    return "\n".join(out)

