from __future__ import annotations

WIDTH = 74


def rule(ch="=", width=WIDTH):
    return ch * width


def banner(title, width=WIDTH):
    return "\n".join([rule("=", width), title, rule("=", width)])


def num(x, dash="—"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.1f}"
    return str(x)


def signed(x):
    if x is None:
        return "—"
    return f"{x:+.1f}"


def grid(headers, rows, indent=4, gap=3):
    table = [headers] + rows
    ncol = len(headers)
    widths = [max(len(str(r[i])) for r in table) for i in range(ncol)]
    pad = " " * indent
    sep = " " * gap
    out = []
    for r in table:
        cells = []
        for i in range(ncol):
            s = str(r[i])
            cells.append(s.ljust(widths[i]) if i == 0 else s.rjust(widths[i]))
        out.append((pad + sep.join(cells)).rstrip())
    return out


def section(number, title):
    return f"{number}. {title}"


def footnotes(lines, indent=4):
    pad = " " * indent
    return [pad + ln for ln in lines]
