from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))

import board
import dataset
from engram.core.memory import Memory
from engram.embeddings import get_embedder

K = 5


def build_store(db, emb):
    mem = Memory(db, embedder=emb, origin_tool="ingest")
    for agent, p in dataset.PROFILES.items():
        mem.register_profile(agent, persona=p["persona"], domain=p["domain"],
                             scope_tags=p["scope"])
    for f in dataset.FACTS:
        mem.save(f["text"], domain=f["domain"])
    return mem


def value_map():
    return {f["text"].strip(): f["id"] for f in dataset.FACTS}


def served(hits, vmap):
    out = []
    for h in hits:
        fid = vmap.get((h.get("value") or "").strip())
        if fid:
            out.append((fid, h.get("domain")))
    return out


def flat(mem, case):
    return mem.search(case["query"], k=K)


def conditioned(mem, case):
    return mem.search(case["query"], agent=case["agent"],
                      task_tags=case.get("task_tags"), k=K)


def evaluate(mem, strat, vmap):
    tp = served_tot = gold_tot = leaks = 0
    ab_correct = ab_total = 0
    for case in dataset.CASES:
        hits = strat(mem, case)
        rows = served(hits, vmap)
        got = {fid for fid, _ in rows}
        agent_domain = dataset.PROFILES[case["agent"]]["domain"]
        for _, dom in rows:
            if dom and dom != agent_domain:
                leaks += 1
        if case["kind"] == "abstain":
            ab_total += 1
            if not hits:
                ab_correct += 1
        else:
            gold = set(case["gold"])
            tp += len(got & gold)
            served_tot += len(got)
            gold_tot += len(gold)
    precision = tp / served_tot if served_tot else 0.0
    recall = tp / gold_tot if gold_tot else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    abstain = ab_correct / ab_total if ab_total else 0.0
    return precision, recall, f1, leaks, f"{ab_correct}/{ab_total}"


def strategy(mem, kind, vmap):
    if kind == "flat":
        return evaluate(mem, flat, vmap)
    os.environ["ENGRAM_SEMANTIC_LANES"] = "off" if kind == "lexical" else "on"
    result = evaluate(mem, conditioned, vmap)
    os.environ.pop("ENGRAM_SEMANTIC_LANES", None)
    return result


def render(rows, engine, n_facts):
    lines = [board.banner("CONDITIONED SERVING  —  MEASURED HERE"), ""]
    lines.append(board.section(1, f"which memory does each agent get? ({engine}, "
                                  f"{n_facts} facts, 3 domains, {len(dataset.CASES)} probes, k={K})"))
    lines.append("")
    header = ["strategy", "precision", "recall", "F1", "leaks", "abstain"]
    grid_rows = []
    for name, (p, r, f1, leaks, ab) in rows:
        grid_rows.append([name, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}", str(leaks), ab])
    lines += board.grid(header, grid_rows)
    lines.append("")
    lines += board.footnotes([
        "flat top-k        : one shared pile, no conditioning (the common baseline)",
        "engram lexical    : lane = task_tags ∩ scope_tags (keyword match only)",
        "engram semantic ◄ : lane by embedding proximity + keyword, RRF-fused",
        "",
        "leaks   = memories served from another agent's domain (lower is better)",
        "abstain = out-of-lane queries correctly served nothing (higher is better)",
        "same store, same probes; the only change is how a lane is decided.",
    ])
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    emb = get_embedder()
    engine = getattr(emb, "model_name", type(emb).__name__)
    calibrated = getattr(emb, "calibrated", True)
    db = os.path.join(tempfile.mkdtemp(prefix="engram-cond-"), "bench.db")
    mem = build_store(db, emb)
    stored = len(mem.all_current())
    vmap = value_map()
    rows = [
        ("flat top-k", strategy(mem, "flat", vmap)),
        ("engram lexical lanes", strategy(mem, "lexical", vmap)),
        ("engram semantic lanes ◄", strategy(mem, "semantic", vmap)),
    ]
    mem.close()
    if not calibrated:
        print("WARNING: embedder is the hash stub (not calibrated); the semantic "
              "lane is inactive and matches the lexical row. Install fastembed for "
              "a real comparison.\n")
    print(render(rows, engine, stored))
    if stored != len(dataset.FACTS):
        print(f"\nnote: {len(dataset.FACTS)} facts defined, {stored} stored "
              "(consolidation merged some).")


if __name__ == "__main__":
    main()
