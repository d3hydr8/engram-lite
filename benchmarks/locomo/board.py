from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import board

SYSTEMS = [
    {"name": "Full-context (no memory)", "overall": 72.9, "single": None,
     "multi": None, "temporal": None, "open": None, "kind": "baseline"},
    {"name": "Mem0-graph", "overall": 68.4, "single": 65.7, "multi": 47.2,
     "temporal": 58.1, "open": 75.7, "kind": "competitor"},
    {"name": "engram-lite", "overall": 68.3, "single": 74.8, "multi": 55.3,
     "temporal": 68.5, "open": 49.0, "kind": "ours"},
    {"name": "Mem0", "overall": 66.9, "single": 67.1, "multi": 51.2,
     "temporal": 55.5, "open": 72.9, "kind": "competitor"},
    {"name": "Zep", "overall": 66.0, "single": 61.7, "multi": 41.4,
     "temporal": 49.3, "open": 76.6, "kind": "competitor", "note": "*"},
    {"name": "LangMem", "overall": 58.1, "single": 62.2, "multi": 47.9,
     "temporal": 23.4, "open": 71.1, "kind": "competitor"},
    {"name": "OpenAI memory", "overall": 52.9, "single": 63.8, "multi": 42.9,
     "temporal": 21.7, "open": 62.3, "kind": "competitor"},
]

PROGRESSION = [
    ("run 1", "strict internal harness", 39.8),
    ("run 2", "mem0 protocol parity", 53.2),
    ("run 3", "+ product changes", 66.9),
    ("run 4", "+ bge-base embedder", 68.3),
]

ADVERSARIAL = {"j_bge_base": 64.6, "j_bge_small": 64.3, "f1": 0.638}

SOURCES = [
    "competitor + full-context rows: mem0 ECAI-2025 (arXiv:2504.19413),",
    "  gpt-4o-mini answerer, LLM judge; all 26 figures verified against the paper.",
    "engram-lite rows: this harness, claude-haiku-4-5 answerer + judge (disclosed",
    "  difference); zero LLM calls to build the memory ($0, ~2 min, local).",
    "LoCoMo: Maharana et al. (arXiv:2402.17753, ACL 2024), cats 1-4 = 1,540 Q,",
    "  adversarial = 446 Q, CC BY-NC 4.0.",
    "* Zep disputed: mem0's paper lists 66.0; mem0's corrected re-run 58.44;",
    "  Zep self-reports 84 (75.14 after disputing config). Vendor numbers vary.",
    "mem0's newer 92.5 figure uses a different protocol (gpt-5 judge, top-200,",
    "  partial credit) and is not comparable to the paper's 66.9 or the rows here.",
]


def _row(s):
    tag = " ◄" if s["kind"] == "ours" else (s.get("note", "") and f' {s["note"]}')
    name = s["name"] + (tag or "")
    return [name, board.num(s["overall"]), board.num(s["single"]),
            board.num(s["multi"]), board.num(s["temporal"]), board.num(s["open"])]


def _find(name):
    return next(s for s in SYSTEMS if s["name"] == name)


def render():
    ours = _find("engram-lite")
    memo = _find("Mem0")
    graph = _find("Mem0-graph")

    lines = [board.banner("ENGRAM-LITE  —  LOCOMO LEADERBOARD"), ""]

    lines.append(board.section(1, "LoCoMo J — mem0's protocol, cats 1-4, 1,540 questions"))
    lines.append("")
    header = ["", "OVERALL", "single-hop", "multi-hop", "temporal", "open-domain"]
    lines += board.grid(header, [_row(s) for s in SYSTEMS])
    lines.append("")
    cats = ["single", "multi", "temporal", "open"]
    dvs_memo = ["engram vs Mem0", board.signed(ours["overall"] - memo["overall"])] + \
        [board.signed(ours[c] - memo[c]) for c in cats]
    lines += board.grid(["", "OVERALL", "single-hop", "multi-hop", "temporal", "open-domain"],
                        [dvs_memo])
    lines += board.footnotes([
        f"engram vs Mem0-graph: {board.signed(ours['overall'] - graph['overall'])} overall "
        "(within run-to-run noise — a statistical tie)."])
    lines.append("")

    lines.append(board.section(2, "engram-lite J progression (same 1,540 questions each run)"))
    lines.append("")
    prog_rows = []
    prev = None
    for tag, note, j in PROGRESSION:
        delta = "" if prev is None else board.signed(j - prev)
        marker = "  ◄ now" if (tag, note, j) == PROGRESSION[-1] else ""
        prog_rows.append([tag, note, board.num(j), delta + marker])
        prev = j
    lines += board.grid(["", "", "J", ""], prog_rows)
    lines.append("")

    lines.append(board.section(3, "adversarial / abstention (446 trick questions — mem0 excludes these)"))
    lines.append("")
    lines += board.footnotes([
        f"engram-lite  J {ADVERSARIAL['j_bge_base']} (bge-base) · "
        f"{ADVERSARIAL['j_bge_small']} (bge-small) · F1 {ADVERSARIAL['f1']}"
        "        mem0: not reported",
        "correct answer = honest abstention; engram serves nothing when it should."])
    lines.append("")

    lines.append(board.rule("-"))
    lines += board.footnotes(SOURCES)
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(render())
